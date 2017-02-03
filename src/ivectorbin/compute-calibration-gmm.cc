// ivectorbin/compute-calibration-gmm.cc

// Copyright 2016  David Snyder

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/model-common.h"
#include "gmm/diag-gmm.h"
#include "gmm/mle-diag-gmm.h"

namespace kaldi {

// We initialize the GMM parameters by setting the variance to the global
// variance of the features, and the means to distinct randomly chosen frames.
void InitGmmFromRandomFrames(const MatrixBase<BaseFloat> &feats, DiagGmm *gmm) {
  int32 num_gauss = gmm->NumGauss(), num_frames = feats.NumRows(),
      dim = feats.NumCols();
  KALDI_ASSERT(num_frames >= 10 * num_gauss && "Too few frames to train on");
  Vector<double> mean(dim), var(dim);
  for (int32 i = 0; i < num_frames; i++) {
    mean.AddVec(1.0 / num_frames, feats.Row(i));
    var.AddVec2(1.0 / num_frames, feats.Row(i));
  }
  var.AddVec2(-1.0, mean);
  if (var.Max() <= 0.0)
    KALDI_ERR << "Features do not have positive variance " << var;
  
  DiagGmmNormal gmm_normal(*gmm);

  std::set<int32> used_frames;
  for (int32 g = 0; g < num_gauss; g++) {
    int32 random_frame = RandInt(0, num_frames - 1);
    while (used_frames.count(random_frame) != 0)
      random_frame = RandInt(0, num_frames - 1);
    used_frames.insert(random_frame);
    gmm_normal.weights_(g) = 1.0 / num_gauss;
    gmm_normal.means_.Row(g).CopyFromVec(feats.Row(random_frame));
    gmm_normal.vars_.Row(g).CopyFromVec(var);
  }
  gmm->CopyFromNormal(gmm_normal);
  gmm->ComputeGconsts();
}

void MleDiagGmmSharedVarsUpdate(const MleDiagGmmOptions &config,
                                const AccumDiagGmm &diag_gmm_acc,
                                GmmFlagsType flags,
                                DiagGmm *gmm,
                                BaseFloat *obj_change_out,
                                BaseFloat *count_out,
                                int32 *floored_elements_out = NULL,
                                int32 *floored_gaussians_out = NULL,
                                int32 *removed_gaussians_out = NULL) {
  KALDI_ASSERT(gmm != NULL);

  if (flags & ~diag_gmm_acc.Flags())
    KALDI_ERR << "Flags in argument do not match the active accumulators";

  KALDI_ASSERT(diag_gmm_acc.NumGauss() == gmm->NumGauss() &&
               diag_gmm_acc.Dim() == gmm->Dim());

  int32 num_gauss = gmm->NumGauss();
  double occ_sum = diag_gmm_acc.occupancy().Sum();

  int32 elements_floored = 0, gauss_floored = 0;
  
  // remember old objective value
  gmm->ComputeGconsts();
  BaseFloat obj_old = MlObjective(*gmm, diag_gmm_acc);

  // First get the gmm in "normal" representation (not the exponential-model
  // form).
  DiagGmmNormal ngmm(*gmm);

  Vector<double> shared_var(gmm->Dim());

  std::vector<int32> to_remove;
  for (int32 i = 0; i < num_gauss; i++) {
    double occ = diag_gmm_acc.occupancy()(i);
    double prob;
    if (occ_sum > 0.0)
      prob = occ / occ_sum;
    else
      prob = 1.0 / num_gauss;

    if (occ > static_cast<double>(config.min_gaussian_occupancy)
        && prob > static_cast<double>(config.min_gaussian_weight)) {
      
      ngmm.weights_(i) = prob;
      
      // copy old mean for later normalizations
      Vector<double> old_mean(ngmm.means_.Row(i));
      
      // update mean, then variance, as far as there are accumulators 
      if (diag_gmm_acc.Flags() & (kGmmMeans|kGmmVariances)) {
        Vector<double> mean(diag_gmm_acc.mean_accumulator().Row(i));
        mean.Scale(1.0 / occ);
        // transfer to estimate
        ngmm.means_.CopyRowFromVec(mean, i);
      }
      
      if (diag_gmm_acc.Flags() & kGmmVariances) {
        KALDI_ASSERT(diag_gmm_acc.Flags() & kGmmMeans);
        Vector<double> var(diag_gmm_acc.variance_accumulator().Row(i));
        var.Scale(1.0 / occ);
        var.AddVec2(-1.0, ngmm.means_.Row(i));  // subtract squared means.
        
        // if we intend to only update the variances, we need to compensate by 
        // adding the difference between the new and old mean
        if (!(flags & kGmmMeans)) {
          old_mean.AddVec(-1.0, ngmm.means_.Row(i));
          var.AddVec2(1.0, old_mean);
        }
        shared_var.AddVec(occ, var);
      }
    } else {  // Insufficient occupancy.
      if (config.remove_low_count_gaussians &&
          static_cast<int32>(to_remove.size()) < num_gauss-1) {
        // remove the component, unless it is the last one.
        KALDI_WARN << "Too little data - removing Gaussian (weight "
                   << std::fixed << prob
                   << ", occupation count " << std::fixed << diag_gmm_acc.occupancy()(i)
                   << ", vector size " << gmm->Dim() << ")";
        to_remove.push_back(i);
      } else {
        KALDI_WARN << "Gaussian has too little data but not removing it because"
                   << (config.remove_low_count_gaussians ?
                       " it is the last Gaussian: i = "
                       : " remove-low-count-gaussians == false: g = ") << i
                   << ", occ = " << diag_gmm_acc.occupancy()(i) << ", weight = " << prob;
        ngmm.weights_(i) =
            std::max(prob, static_cast<double>(config.min_gaussian_weight));
      }
    }
  }
        
  if (diag_gmm_acc.Flags() & kGmmVariances) {
    int32 floored;
    if (config.variance_floor_vector.Dim() != 0) {
      floored = shared_var.ApplyFloor(config.variance_floor_vector);
    } else {
      floored = shared_var.ApplyFloor(config.min_variance);
    }
    if (floored != 0) {
      elements_floored += floored;
      gauss_floored++;
    }

    shared_var.Scale(1.0 / occ_sum);
    for (int32 i = 0; i < num_gauss; i++) {
      ngmm.vars_.CopyRowFromVec(shared_var, i);
    }
  }
  
  // copy to natural representation according to flags
  ngmm.CopyToDiagGmm(gmm, flags);

  gmm->ComputeGconsts();  // or MlObjective will fail.
  BaseFloat obj_new = MlObjective(*gmm, diag_gmm_acc);
  
  if (obj_change_out) 
    *obj_change_out = (obj_new - obj_old);
  if (count_out) *count_out = occ_sum;
  if (floored_elements_out) *floored_elements_out = elements_floored;
  if (floored_gaussians_out) *floored_gaussians_out = gauss_floored;
  
  if (to_remove.size() > 0) {
    gmm->RemoveComponents(to_remove, true /*renormalize weights*/);
    gmm->ComputeGconsts();
  }
  if (removed_gaussians_out != NULL) *removed_gaussians_out = to_remove.size();

  if (gauss_floored > 0)
    KALDI_VLOG(2) << gauss_floored << " variances floored in " << gauss_floored
                  << " Gaussians.";
}

void TrainOneIter(const MatrixBase<BaseFloat> &feats,
                  const MleDiagGmmOptions &gmm_opts,
                  int32 iter,
                  int32 num_threads,
                  bool share_covars,
                  DiagGmm *gmm) {
  AccumDiagGmm gmm_acc(*gmm, kGmmAll);

  Vector<BaseFloat> frame_weights(feats.NumRows(), kUndefined);
  frame_weights.Set(1.0);

  double tot_like;
  tot_like = gmm_acc.AccumulateFromDiagMultiThreaded(*gmm, feats, frame_weights,
                                                     num_threads);

  KALDI_LOG << "Likelihood per frame on iteration " << iter
            << " was " << (tot_like / feats.NumRows()) << " over "
            << feats.NumRows() << " frames.";
  
  BaseFloat objf_change, count;
  if (share_covars) {
    MleDiagGmmSharedVarsUpdate(gmm_opts, gmm_acc, kGmmAll, gmm, 
                               &objf_change, &count);
  } else {
    MleDiagGmmUpdate(gmm_opts, gmm_acc, kGmmAll, gmm, &objf_change, &count);
  }

  KALDI_LOG << "Objective-function change on iteration " << iter << " was "
            << (objf_change / count) << " over " << count << " frames.";
}

void TrainGmm(const MatrixBase<BaseFloat> &feats, 
              const MleDiagGmmOptions &gmm_opts,
              int32 num_gauss, int32 num_gauss_init, int32 num_iters,
              int32 num_threads, bool share_covars, DiagGmm *gmm) {
  KALDI_LOG << "Initializing GMM means from random frames to "
            << num_gauss_init << " Gaussians.";
  InitGmmFromRandomFrames(feats, gmm);

  // we'll increase the #Gaussians by splitting,
  // till halfway through training.
  int32 cur_num_gauss = num_gauss_init,
      gauss_inc = (num_gauss - num_gauss_init) / (num_iters / 2);
      
  for (int32 iter = 0; iter < num_iters; iter++) {
    TrainOneIter(feats, gmm_opts, iter, num_threads, share_covars, gmm);

    int32 next_num_gauss = std::min(num_gauss, cur_num_gauss + gauss_inc);
    if (next_num_gauss > gmm->NumGauss()) {
      KALDI_LOG << "Splitting to " << next_num_gauss << " Gaussians.";
      gmm->Split(next_num_gauss, 0.1);
      cur_num_gauss = next_num_gauss;
    }
  }
}

} // namespace kaldi

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Computes a calibration threshold from scores (e.g., PLDA scores)."
      "Generally, the scores are the result of a comparison between two"
      "iVectors.  This is typically used to find the stopping criteria for"
      "agglomerative clustering."
      "Usage: compute-calibration [options] <scores-rspecifier> "
      "<calibration-wxfilename>\n"
      "e.g.: \n"
      " compute-calibration ark:scores.ark threshold.txt\n";

    ParseOptions po(usage);
    bool read_matrices = true;
    bool ignore_diagonals = false;
    bool share_covars = false;
    int32 num_points = 0;

    po.Register("read-matrices", &read_matrices, "If true, read scores as"
      "matrices, probably output from ivector-plda-scoring-dense");
    po.Register("ignore-diagonals", &ignore_diagonals, "If true, the "
                "diagonals (representing the same segments) will not be "
                "considered for calibration.");
    po.Register("num-points", &num_points, "If specified, use a sample of "
                "these many points.");
    po.Register("share-covars", &share_covars, "If true, then the variances "
                "of the Gaussian components are tied.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string scores_rspecifier = po.GetArg(1),
      calibration_wxfilename = po.GetArg(2);
    BaseFloat mean = 0.0;
    int32 num_done = 0,
      num_err = 0;
    Output output(calibration_wxfilename, false);
    {
      SequentialBaseFloatMatrixReader scores_reader(scores_rspecifier);
      for (; !scores_reader.Done(); scores_reader.Next()) {
        std::string utt = scores_reader.Key();
        const Matrix<BaseFloat> &scores = scores_reader.Value();
        if (scores.NumRows() <= 2 && scores.NumCols() <= 2) {
          KALDI_WARN << "Too few scores in " << utt << " to cluster";
          num_err++;
          continue;
        }

        Matrix<BaseFloat> feats;
        
        int32 this_num_points = num_points;
        if (num_points == 0) {
          this_num_points = scores.NumRows() * scores.NumCols();
        }
        feats.Resize(this_num_points, 1);

        int64 num_read = 0; 
        for (int32 i = 0; i < scores.NumRows(); i++) {
          for (int32 j = 0; j < scores.NumCols(); j++) {
            if (!ignore_diagonals || i != j) {
              num_read++;
              if (num_read >= this_num_points) {
                BaseFloat keep_prob = this_num_points 
                  / static_cast<BaseFloat>(num_read);
                if (WithProb(keep_prob)) {
                  int32 p = RandInt(0, this_num_points-1);
                  feats(p, 0) =  scores(i, j);
                }
              } else {
                feats(num_read - 1, 0) = scores(i, j);
              }
            }
          }
        }
        KALDI_ASSERT(num_read > 0);
        
        if (num_read < this_num_points) {
          KALDI_WARN << "For utterance " << utt << ", "
                     << "number of points read " << num_read << " was less than "
                     << "target number " << this_num_points << ", using all we read.";
          feats.Resize(num_read, 1, kCopyData);
        } else {
          BaseFloat percent = this_num_points * 100.0 / num_read;
          KALDI_LOG << "For utterance " << utt << ", "
                    << "kept " << this_num_points << " out of " << num_read
                    << " input points = " << percent << "%.";
        }

        DiagGmm gmm(2, 1);
        MleDiagGmmOptions gmm_opts;
        TrainGmm(feats, gmm_opts, 2, 2, 20, 1, share_covars, &gmm);

        mean = (gmm.means_invvars()(0, 0) + gmm.means_invvars()(1, 0)) 
                / (gmm.inv_vars()(0, 0) + gmm.inv_vars()(1, 0));

        BaseFloat this_mean1 = gmm.means_invvars()(0, 0) / gmm.inv_vars()(0, 0),
                  this_mean2 = gmm.means_invvars()(1, 0) / gmm.inv_vars()(1, 0);
        KALDI_LOG << "For key " << utt << " the means of the Gaussians are "
                  << this_mean1 << " and " << this_mean2 
                  << "; the variances are " << 1 / gmm.inv_vars()(0, 0)
                  << " and " << 1 /  gmm.inv_vars()(1, 0);

        num_done++;
      }
      mean = mean / (num_done);
    } 
    output.Stream() << mean;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

