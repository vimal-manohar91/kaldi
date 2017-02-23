// ivectorbin/compute-calibration-gmm-supervised.cc

// Copyright 2016  David Snyder
//           2017  Vimal Manohar

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

void MapDiagGmmSharedVarsUpdate(const MapDiagGmmOptions &config,
                                const AccumDiagGmm &diag_gmm_acc,
                                GmmFlagsType flags,
                                DiagGmm *gmm,
                                BaseFloat *obj_change_out,
                                BaseFloat *count_out) {
  KALDI_ASSERT(gmm != NULL);

  if (flags & ~diag_gmm_acc.Flags())
    KALDI_ERR << "Flags in argument do not match the active accumulators";
  
  KALDI_ASSERT(diag_gmm_acc.NumGauss() == gmm->NumGauss() &&
               diag_gmm_acc.Dim() == gmm->Dim());
  
  int32 num_gauss = gmm->NumGauss();
  double occ_sum = diag_gmm_acc.occupancy().Sum();
  
  // remember the old objective function value
  gmm->ComputeGconsts();
  BaseFloat obj_old = MlObjective(*gmm, diag_gmm_acc);

  // allocate the gmm in normal representation; all parameters of this will be 
  // updated, but only the flagged ones will be transferred back to gmm
  DiagGmmNormal ngmm(*gmm);
  Vector<double> shared_var(gmm->Dim());

  for (int32 i = 0; i < num_gauss; i++) {
    double occ = diag_gmm_acc.occupancy()(i);

    // First update the weight.  The weight_tau is a tau for the
    // whole state.
    ngmm.weights_(i) = (occ + ngmm.weights_(i) * config.weight_tau) /
        (occ_sum + config.weight_tau);


    if (occ > 0.0 && (flags & kGmmMeans)) {
      // Update the Gaussian mean.
      Vector<double> old_mean(ngmm.means_.Row(i));
      Vector<double> mean(diag_gmm_acc.mean_accumulator().Row(i));
      mean.Scale(1.0 / (occ + config.mean_tau));
      mean.AddVec(config.mean_tau / (occ + config.mean_tau), old_mean);
      ngmm.means_.CopyRowFromVec(mean, i);
    }
    
    if (occ > 0.0 && (flags & kGmmVariances)) {
      // Computing the variance around the updated mean; this is:
      // E( (x - mu)^2 ) = E( x^2 - 2 x mu + mu^2 ) =
      // E(x^2) + mu^2 - 2 mu E(x).
      Vector<double> old_var(ngmm.vars_.Row(i));
      Vector<double> var(diag_gmm_acc.variance_accumulator().Row(i));
      var.Scale(1.0 / occ);
      var.AddVec2(1.0, ngmm.means_.Row(i));
      SubVector<double> mean_acc(diag_gmm_acc.mean_accumulator(), i),
          mean(ngmm.means_, i);
      var.AddVecVec(-2.0 / occ, mean_acc, mean, 1.0);
      
      // now var is E(x^2) + m^2 - 2 mu E(x).
      // Next we do the appropriate weighting usnig the tau value.
      var.Scale(occ / (config.variance_tau + occ));
      var.AddVec(config.variance_tau / (config.variance_tau + occ), old_var);
      // This is what would be if variances are not shared:
      // ngmm.vars_.Row(i).CopyFromVec(var);
      
      shared_var.AddVec(occ + config.variance_tau, var);
    }
  }

  shared_var.Scale(1.0 / (occ_sum + num_gauss * config.variance_tau));
  for (int32 i = 0; i < num_gauss; i++) {
    ngmm.vars_.CopyRowFromVec(shared_var, i);
  }
  
  // Copy to natural/exponential representation.
  ngmm.CopyToDiagGmm(gmm, flags);

  gmm->ComputeGconsts();  // or MlObjective will fail.
  BaseFloat obj_new = MlObjective(*gmm, diag_gmm_acc);
  
  if (obj_change_out) 
    *obj_change_out = (obj_new - obj_old);
  
  if (count_out) *count_out = occ_sum;
}

void TrainOneIter(const MatrixBase<BaseFloat> &feats,
                  const MapDiagGmmOptions &gmm_opts,
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
    MapDiagGmmSharedVarsUpdate(gmm_opts, gmm_acc, kGmmAll, gmm, 
                               &objf_change, &count);
  } else {
    MapDiagGmmUpdate(gmm_opts, gmm_acc, kGmmAll, gmm, &objf_change, &count);
  }

  KALDI_LOG << "Objective-function change on iteration " << iter << " was "
            << (objf_change / count) << " over " << count << " frames.";
}


BaseFloat ComputeThreshold(const MatrixBase<BaseFloat> &feats,
                           const MapDiagGmmOptions &gmm_opts,
                           BaseFloat same_class_prior,
                           BaseFloat same_class_mean,
                           BaseFloat different_class_mean,
                           BaseFloat same_class_variance,
                           BaseFloat different_class_variance,
                           bool share_covars,
                           const std::string *utt = NULL) {
  DiagGmm gmm(2, 1);

  DiagGmmNormal ngmm;
  ngmm.Resize(2, 1);

  ngmm.weights_(1) = same_class_prior;
  ngmm.weights_(0) = 1.0 - same_class_prior;

  ngmm.means_(1, 0) = same_class_mean;
  ngmm.means_(0, 0) = different_class_mean;

  ngmm.vars_(1, 0) = same_class_variance;
  ngmm.vars_(0, 0) = different_class_variance;

  gmm.CopyFromNormal(ngmm);
  gmm.ComputeGconsts();

  for (int32 iter = 0; iter < 20; iter++) {
    TrainOneIter(feats, gmm_opts, iter, 1, share_covars, &gmm);
  }

  BaseFloat mean = (gmm.means_invvars()(0, 0) + gmm.means_invvars()(1, 0)) 
                    / (gmm.inv_vars()(0, 0) + gmm.inv_vars()(1, 0));
  
  BaseFloat this_mean1 = gmm.means_invvars()(0, 0) / gmm.inv_vars()(0, 0),
            this_mean2 = gmm.means_invvars()(1, 0) / gmm.inv_vars()(1, 0);

  if (utt) {
    KALDI_LOG << "For key " << *utt << " the means of the Gaussians are "
              << this_mean1 << " and " << this_mean2 
              << "; the variances are " << 1 / gmm.inv_vars()(0, 0)
              << " and " << 1 /  gmm.inv_vars()(1, 0)
              << "; the weights are " << gmm.weights()(0) << " and " 
              << gmm.weights()(1);
  } else {
    KALDI_LOG << "The means of the Gaussians are "
              << this_mean1 << " and " << this_mean2 
              << "; the variances are " << 1 / gmm.inv_vars()(0, 0)
              << " and " << 1 /  gmm.inv_vars()(1, 0)
              << "; the weights are " << gmm.weights()(0) << " and " 
              << gmm.weights()(1);
  }
  return mean;
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
      "Usage: compute-calibration-gmm-supervised [options] <scores-rspecifier> "
      "<threshold-wspecifier|threshold-wxfilename>\n"
      "e.g.: \n"
      " compute-calibration ark:scores.ark threshold.txt\n";

    ParseOptions po(usage);
    bool read_matrices = true;
    bool ignore_diagonals = false;
    bool share_covars = false;
    int32 num_points = 0;
    int32 local_window = -1;
    BaseFloat same_class_prior = 0.5;
    BaseFloat same_class_mean = 1.0;
    BaseFloat different_class_mean = -1.0;
    BaseFloat same_class_variance = 1.0;
    BaseFloat different_class_variance = 4.0;
    
    po.Register("read-matrices", &read_matrices, "If true, read scores as"
      "matrices, probably output from ivector-plda-scoring-dense");
    po.Register("ignore-diagonals", &ignore_diagonals, "If true, the "
                "diagonals (representing the same segments) will not be "
                "considered for calibration.");
    po.Register("select-local-window", &local_window, "If specified, "
                "select point only from a local window of these many points.");
    po.Register("num-points", &num_points, "If specified, use a sample of "
                "these many points.");
    po.Register("share-covars", &share_covars, "If true, then the variances "
                "of the Gaussian components are tied.");
    po.Register("same-class-prior", &same_class_prior, "The prior for the "
                "Gaussian corresponding to the same class.");
    po.Register("same-class-mean", &same_class_mean, "The mean for the "
                "same class scores.");
    po.Register("different-class-mean", &different_class_mean, 
                "The mean for the different class scores.");
    po.Register("same-class-variance", &same_class_variance, 
                "The variance of the same class scores.");
    po.Register("different-class-variance", &different_class_variance,
                "The variance of the different class scores.");

    MapDiagGmmOptions gmm_opts;
    gmm_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string scores_rspecifier = po.GetArg(1),
      threshold_out_fn = po.GetArg(2);
    int32 num_done = 0,
      num_err = 0;

    bool out_is_wspecifier = false;
    Matrix<BaseFloat> feats;
    
    BaseFloatWriter *threshold_writer;

    if (ClassifyWspecifier(threshold_out_fn, NULL, NULL, NULL)) {
      threshold_writer = new BaseFloatWriter(threshold_out_fn);
      out_is_wspecifier = true;
    } else {
      if (num_points == 0) {
        KALDI_ERR << "--num-points must be specified when output is "
                  << "not archive.";
      }
    }
      
    if (num_points > 0) 
      feats.Resize(num_points, 1);

    SequentialBaseFloatMatrixReader scores_reader(scores_rspecifier);
    int64 num_read = 0; 
  
    
    for (; !scores_reader.Done(); scores_reader.Next()) {
      const std::string &utt = scores_reader.Key();
      const Matrix<BaseFloat> &scores = scores_reader.Value();
      if (scores.NumRows() <= 2 && scores.NumCols() <= 2) {
        KALDI_WARN << "Too few scores in " << utt << " to cluster";
        num_err++;
        continue;
      }

      int32 this_num_points = num_points;
      if (out_is_wspecifier) {
        num_read = 0;
        if (num_points == 0) {
          this_num_points = scores.NumRows() * scores.NumCols();
        }
        feats.Resize(this_num_points, 1, kSetZero);
      }

      for (int32 i = 0; i < scores.NumRows(); i++) {
        for (int32 j = 0; j < scores.NumCols(); j++) {
          if (local_window > 0) {
            if (std::abs(i - j) > local_window) continue;
          }
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

      if (out_is_wspecifier) {
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

        BaseFloat mean = ComputeThreshold(
            feats, gmm_opts, same_class_prior, 
            same_class_mean, different_class_mean,
            same_class_variance, different_class_variance,
            share_covars, &utt);

        threshold_writer->Write(utt, mean);
      } 
    } 
      
    if (!out_is_wspecifier) {
      KALDI_ASSERT(num_read > 0);
      
      if (num_read < num_points) {
        KALDI_WARN << "Number of points read " << num_read << " was less than "
                   << "target number " << num_points << ", using all we read.";
        feats.Resize(num_read, 1, kCopyData);
      } else {
        BaseFloat percent = num_points * 100.0 / num_read;
        KALDI_LOG << "Kept " << num_points << " out of " << num_read
                  << " input points = " << percent << "%.";
      }

      BaseFloat mean = ComputeThreshold(
          feats, gmm_opts, same_class_prior, 
          same_class_mean, different_class_mean,
          same_class_variance, different_class_variance,
          share_covars);

      Output ko(threshold_out_fn, false);
      ko.Stream() << mean;
    } else {
      delete threshold_writer;
    }
    
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

