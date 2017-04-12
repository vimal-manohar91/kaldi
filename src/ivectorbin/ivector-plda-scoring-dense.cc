// ivectorbin/ivector-plda-scoring-dense.cc

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
#include "util/stl-utils.h"
#include "ivector/plda.h"
#include "ivector/transform.h"

namespace kaldi {

void CheckPldaObjf() {
  Vector<double> vec1(2);
  Vector<double> vec2(2);
  vec1(0) = 1;
  vec2(0) = 2;
  vec1(1) = 2;
  vec2(1) = 1;

  Vector<double> psi(2);
  psi(0) = 1;
  psi(1) = 0.5;

  double ans = 0.0;
  for (int32 i = 0; i < 2; i++) {
    double mean = 0.5 * (vec1(i) + vec2(i));         // mean = [1, 2]
    ans += 0.5 * Log(psi(i) + 1) - 0.5 * Log((2 * psi(i) + 1) /(psi(i) + 1)) // 0.5 Log(2) - 0.5 * Log(1.5/2) ~ 0.49
      - 0.5 * mean * mean / (psi(i) + 1.0 / 2)           // -[1 / 3, 4 / 3]
      + 0.5 * vec1(i) * vec1(i) / (psi(i) + 1.0)     // 0.5 * [1, 4] / 2
      + 0.5 * vec2(i) * vec2(i) / (psi(i) + 1.0)     // 0.5 * [1, 4] / 2
      - 0.5 * (vec1(i) - mean) * (vec1(i) - mean)    // 0
      - 0.5 * (vec2(i) - mean) * (vec2(i) - mean);   // 0
  }

  // Log(2) - 0.5 * Log(1.5) - 1/3 + 0.5 / 2 + 0.5 / 2
  // + Log(2) - 0.5 * Log(1.5) - 4/3 + 0.5 * 4 / 2 + 0.5 * 4 / 2
  // = 2 * Log(2) - Log(1.5) - 5 / 3 + 2.5 / 2 + 2.5 / 2
  // = 2 * Log(2) - Log(1.5) - 5 / 3 + 5 / 2
  // = 2 * Log(2) - Log(1.5) + 5 / 6
  
  double llr =0.0;
  {
    int32 dim = 2;
    int32 n = 1;
    double loglike_given_class, loglike_without_class;
    { // work out loglike_given_class.
      // "mean" will be the mean of the distribution if it comes from the
      // training example.  The mean is \frac{n \Psi}{n \Psi + I} \bar{u}^g
      // "variance" will be the variance of that distribution, equal to
      // I + \frac{\Psi}{n\Psi + I}.
      Vector<double> mean(dim, kUndefined);
      Vector<double> variance(dim, kUndefined);
      for (int32 i = 0; i < dim; i++) {
        mean(i) = 1 * psi(i) / (psi(i) + 1.0)
          * vec1(i);  // [0.5, 1]
        variance(i) = 1.0 + psi(i) / (n * psi(i) + 1.0); // 1 + 1 / 2
      }
      double logdet = variance.SumLog();  // 2 * Log(1.5) 
      Vector<double> sqdiff(vec2);        // [1, 2]
      sqdiff.AddVec(-1.0, mean);          // [0.5, 1]
      sqdiff.ApplyPow(2.0);               // [0.25, 1]
      variance.InvertElements();          // 1/1.5
      loglike_given_class = -0.5 * (logdet + M_LOG_2PI * dim +
          VecVec(sqdiff, variance));      // -0.5 * (2 * Log(1.5) + 1.25 * 2/3)
    }
    { // work out loglike_without_class.  Here the mean is zero and the variance
      // is I + \Psi.
      Vector<double> sqdiff(vec2); // there is no offset.     // [1, 2]
      sqdiff.ApplyPow(2.0);                                   // [1, 4]
      Vector<double> variance(psi);                           // [1, 1]
      variance.Add(1.0); // I + \Psi.                         // [2, 2]
      double logdet = variance.SumLog();                      // 2 * Log(2)
      variance.InvertElements();                              // [0.5, 0.5]
      loglike_without_class = -0.5 * (logdet + M_LOG_2PI * dim +
          VecVec(sqdiff, variance));    // -0.5 * (2 * Log(2) + 1 * 0.5 + 4 * 0.5)
    }
    llr = loglike_given_class - loglike_without_class; 
      // 0.5 * (2 * Log(2) - 2 * Log(1.5) + 2.5 - 5 / 6)
      // Log(2) - Log(1.5) + 5 / 6
  }

  KALDI_ASSERT(kaldi::ApproxEqual(llr, ans));
}

}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Perform PLDA scoring for speaker diarization.  The input reco2utt\n"
      "should be of the form <recording-id> <seg1> <seg2> ... <segN> and\n"
      "there should be one iVector for each segment.  PLDA scoring is\n"
      "performed between all pairs of iVectors in a recording and outputs\n"
      "an archive of score matrices, one for each recording-id.  The rows\n"
      "and columns of the the matrix correspond the sorted order of the\n"
      "segments.\n"
      "Usage: ivector-diarization-plda-scoring [options] <plda> <reco2utt>"
      " <ivectors-rspecifier> <scores-wspecifier>\n"
      "e.g.: \n"
      "  ivector-diarization-plda-scoring plda reco2utt scp:ivectors.scp"
      " ark:scores.ark ark,t:ivectors.1.ark\n";

    ParseOptions po(usage);
    BaseFloat target_energy = 0.5;
    bool apply_logistic = false;
    std::string utt2num_frames_rspecifier;

    PldaConfig plda_config;
    plda_config.Register(&po);

    po.Register("target-energy", &target_energy,
      "Reduce dimensionality of i-vectors using PCA such that this fraction"
      " of the total energy remains.");
    po.Register("apply-logistic", &apply_logistic,
                "If specified, the scores are transformed using a "
                "logistic function.");
    po.Register("utt2num-frames-rspecifier", &utt2num_frames_rspecifier,
                "Weight i-vector by number of frames.");
    KALDI_ASSERT(target_energy <= 1.0);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda_rxfilename = po.GetArg(1),
      reco2utt_rspecifier = po.GetArg(2),
      ivector_rspecifier = po.GetArg(3),
      scores_wspecifier = po.GetArg(4),
      out_reco2utt_wspecifier = po.GetArg(5);

    Plda plda;
    ReadKaldiObject(plda_rxfilename, &plda);

    SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
    RandomAccessBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    BaseFloatMatrixWriter scores_writer(scores_wspecifier);
    TokenVectorWriter out_reco2utt_writer(out_reco2utt_wspecifier);
    RandomAccessInt32Reader utt2num_frames_reader(utt2num_frames_rspecifier);

    int32 num_reco_err = 0, num_reco_done = 0, num_utt_err = 0;
    for (; !reco2utt_reader.Done(); reco2utt_reader.Next()) {
      Plda this_plda(plda);
      const std::string &reco = reco2utt_reader.Key();

      // The uttlist is sorted here and in binaries that use the scores
      // this outputs.  This is to ensure that the segment corresponding
      // to the same rows and columns (of the score matrix) across binaries.
      const std::vector<std::string> &uttlist = reco2utt_reader.Value();
      std::vector<std::string> out_uttlist;

      Matrix<BaseFloat> ivector_mat;
      Vector<BaseFloat> row_weights(uttlist.size());
      row_weights.Set(1.0);

      int32 id = 0;
      for (size_t i = 0; i < uttlist.size(); i++) {
        const std::string &utt = uttlist[i];

        if (!ivector_reader.HasKey(utt)) {
          KALDI_WARN << "No iVector present in input for utterance " << utt;
          num_utt_err++;
          continue;
        }

        const Vector<BaseFloat> &ivector = ivector_reader.Value(utt);

        if (ivector_mat.NumRows() == 0) {
          ivector_mat.Resize(uttlist.size(), ivector.Dim());
        }
      
        if (!utt2num_frames_rspecifier.empty()) {
          if (!utt2num_frames_reader.HasKey(utt)) {
            KALDI_WARN << "No weights present for utterance " << utt
                       << "in rspecifier " << utt2num_frames_rspecifier;
            num_utt_err++;
            continue;
          }

          row_weights(id) = utt2num_frames_reader.Value(utt);
        } else {
          row_weights(id) = 1.0;
        }

        ivector_mat.CopyRowFromVec(ivector, id);
        out_uttlist.push_back(utt);
        id++;
      }

      if (uttlist.size() == 0) {
        KALDI_WARN << "Not producing output for recording " << reco
                   << " since no segments had iVectors";
        num_reco_err++;
        continue;
      }
      ivector_mat.Resize(out_uttlist.size(), ivector_mat.NumCols(), kCopyData);
      row_weights.Resize(out_uttlist.size(), kCopyData);

      Matrix<BaseFloat> ivector_mat_pca, ivector_mat_plda, pca_transform,
        scores(out_uttlist.size(), out_uttlist.size());

      if (EstPca(ivector_mat, row_weights, target_energy, &pca_transform)) {
        // Apply PCA transform to the raw i-vectors.
        ApplyPca(ivector_mat, pca_transform, &ivector_mat_pca);

        // Apply PCA transform to the parameters of the PLDA model.
        this_plda.ApplyTransform(Matrix<double>(pca_transform));

        // Now transform the i-vectors using the reduced PLDA model.
        TransformIvectors(ivector_mat_pca, plda_config, this_plda,
                          &ivector_mat_plda);
        KALDI_LOG << "For recording, " << reco << " retained " 
                  << this_plda.Dim() << " dimensions.";
      } else {
        KALDI_WARN << "Unable to compute conversation dependent PCA for"
                   << " recording " << reco << ".";
        ivector_mat_pca.Resize(ivector_mat.NumRows(), ivector_mat.NumCols());
        ivector_mat_pca.CopyFromMat(ivector_mat);
      }
        
      for (int32 i = 0; i < ivector_mat_plda.NumRows(); i++) {
        for (int32 j = 0; j < ivector_mat_plda.NumRows(); j++) {
          scores(i,j) = this_plda.LogLikelihoodRatio(
              Vector<double>(ivector_mat_plda.Row(i)), 1.0,
              Vector<double>(ivector_mat_plda.Row(j)));

          CheckPldaObjf();
          // Pass the raw PLDA scores through a logistic function
          // so that they are between 0 and 1.
          //scores(i,j) = 1.0
          //  / (1.0 + exp(this_plda.LogLikelihoodRatio(Vector<double>(
          //  ivector_mat_plda.Row(i)), 1.0,
          //  Vector<double>(ivector_mat_plda.Row(j)))));
        }
      }

      if (apply_logistic) {
        scores.Sigmoid(scores);
      }

      scores_writer.Write(reco, scores);
      out_reco2utt_writer.Write(reco, out_uttlist);
      num_reco_done++;
    }
    KALDI_LOG << "Processed " << num_reco_done << " recordings, "
              << num_reco_err << " had errors.";
    return (num_reco_done != 0 ? 0 : 1 );
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
