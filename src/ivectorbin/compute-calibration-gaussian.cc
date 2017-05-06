// ivectorbin/compute-calibration-gaussian.cc

// Copyright 2017  Vimal Manohar

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
#include "tree/clusterable-classes.h"

namespace kaldi {

BaseFloat ComputeThreshold(const ScalarClusterable &sc,
                           BaseFloat threshold_stddev,
                           const std::string *reco = NULL) {
  BaseFloat mean = sc.Mean();
  BaseFloat var = sc.Variance();
  BaseFloat stddev = std::sqrt(var);
  
  BaseFloat threshold = mean + threshold_stddev * stddev;
  
  if (reco) {
    KALDI_LOG << "For key " << *reco << " the mean and stddev of the "
              << "Gaussian is " << mean << " and " << stddev;
  } else {
    KALDI_LOG << "The mean and stddev of the "
              << "Gaussian is " << mean << " and " << stddev;
  }
  return threshold;
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
      "<threshold-wspecifier|threshold-wxfilename>\n"
      "e.g.: \n"
      " compute-calibration ark:scores.ark threshold.txt\n";

    ParseOptions po(usage);
    bool ignore_diagonals = false;
    int32 local_window = -1;
    BaseFloat threshold_stddev = 2.0;
    
    po.Register("ignore-diagonals", &ignore_diagonals, "If true, the "
                "diagonals (representing the same segments) will not be "
                "considered for calibration.");
    po.Register("select-local-window", &local_window, "If specified, "
                "select point only from a local window of these many points.");
    po.Register("threshold-stddev", &threshold_stddev, "Use as threshold, "
                "this factor of standard deviation from the mean.");

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
    
    BaseFloatWriter *threshold_writer;

    if (ClassifyWspecifier(threshold_out_fn, NULL, NULL, NULL)) {
      threshold_writer = new BaseFloatWriter(threshold_out_fn);
      out_is_wspecifier = true;
    } 
      
    SequentialBaseFloatMatrixReader scores_reader(scores_rspecifier);
    
    ScalarClusterable sc;
    for (; !scores_reader.Done(); scores_reader.Next()) {
      const std::string &reco = scores_reader.Key();
      const Matrix<BaseFloat> &scores = scores_reader.Value();
      if (scores.NumRows() <= 2 && scores.NumCols() <= 2) {
        KALDI_WARN << "Too few scores in " << reco << " to cluster";
        num_err++;
        continue;
      }

      ScalarClusterable reco_sc;

      for (int32 i = 0; i < scores.NumRows(); i++) {
        for (int32 j = 0; j < scores.NumCols(); j++) {
          if (local_window > 0) {
            if (std::abs(i - j) > local_window) continue;
          }
          if (!ignore_diagonals || i != j) {
            ScalarClusterable this_sc(scores(i, j));
            reco_sc.Add(this_sc);
          }
        }
      }

      sc.Add(reco_sc);

      if (out_is_wspecifier) {
        BaseFloat thresh = ComputeThreshold(reco_sc, threshold_stddev, &reco);
        threshold_writer->Write(reco, thresh);
      } 
      num_done++;
    } 
      
    if (!out_is_wspecifier) {
      BaseFloat thresh = ComputeThreshold(sc, threshold_stddev);

      Output ko(threshold_out_fn, false);
      ko.Stream() << thresh;
    } else {
      delete threshold_writer;
    }
    
    return (num_done > num_err && num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

