// ivectorbin/compute-calibration.cc

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
#include "tree/cluster-utils.h"
#include "tree/clusterable-classes.h"

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
    int32 num_points = 0;
    int32 local_window = -1;

    po.Register("read-matrices", &read_matrices, "If true, read scores as"
      "matrices, probably output from ivector-plda-scoring-dense");
    po.Register("ignore-diagonals", &ignore_diagonals, "If true, the "
                "diagonals (representing the same segments) will not be "
                "considered for calibration.");
    po.Register("select-local-window", &local_window, "If specified, "
                "select point only from a local window of these many points.");
    po.Register("num-points", &num_points, "If specified, use a sample of "
                "these many points.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string scores_rspecifier = po.GetArg(1),
      calibration_wxfilename = po.GetArg(2);
    ClusterKMeansOptions opts;
    BaseFloat mean = 0.0;
    int32 num_done = 0,
      num_err = 0;
    Output output(calibration_wxfilename, false);
    if (read_matrices) {
      SequentialBaseFloatMatrixReader scores_reader(scores_rspecifier);
      for (; !scores_reader.Done(); scores_reader.Next()) {
        std::string utt = scores_reader.Key();
        const Matrix<BaseFloat> &scores = scores_reader.Value();
        if (scores.NumRows() <= 2 && scores.NumCols() <= 2) {
          KALDI_WARN << "Too few scores in " << utt << " to cluster";
          num_err++;
          continue;
        }
        std::vector<Clusterable*> this_clusterables;
        std::vector<Clusterable*> this_clusters;

        int32 this_num_points = num_points;
        if (num_points > 0) {
          this_clusterables.reserve(num_points);
        } else {
          this_clusterables.reserve(scores.NumRows() * scores.NumCols());
          this_num_points = scores.NumRows() * scores.NumCols();
        }

        this_clusterables.reserve(this_num_points);
        int32 num_current_points = 0;
        for (int32 i = 0; i < scores.NumRows(); i++) {
          for (int32 j = 0; j < scores.NumCols(); j++) {
            if (local_window > 0) {
              if (std::abs(i - j) > local_window) continue;
            }
            if (!ignore_diagonals || i != j) {
              if (num_current_points >= this_num_points) {
                if (WithProb(0.5)) {
                  int32 p = RandInt(0, this_num_points-1);
                  delete this_clusterables[p];
                  this_clusterables[p] = new ScalarClusterable(scores(i, j));
                }
              } else {
                this_clusterables.push_back(new ScalarClusterable(scores(i, j)));
                num_current_points++;
              }
            }
          }
        }

        ClusterKMeans(this_clusterables, 2, &this_clusters, NULL, opts);
        DeletePointers(&this_clusterables);
        BaseFloat this_mean1 = static_cast<ScalarClusterable*>(
          this_clusters[0])->Mean(),
          this_mean2 = static_cast<ScalarClusterable*>(
          this_clusters[1])->Mean();
        KALDI_LOG << "For key " << utt << " the means of the Gaussians are "
                  << this_mean1 << " and " << this_mean2;
        mean += this_mean1 + this_mean2;
        num_done++;
      }
      mean = mean / (2*num_done);
    } else {
      std::vector<Clusterable*> clusterables;
      std::vector<Clusterable*> clusters;
      SequentialBaseFloatReader scores_reader(scores_rspecifier);
      for (; !scores_reader.Done(); scores_reader.Next()) {
        std::string utt = scores_reader.Key();
        const BaseFloat score = scores_reader.Value();
        clusterables.push_back(new ScalarClusterable(score));
        num_done++;
      }
      ClusterKMeans(clusterables, 2, &clusters, NULL, opts);
      DeletePointers(&clusterables);
      BaseFloat mean1 = static_cast<ScalarClusterable*>(clusters[0])->Mean(),
        mean2 = static_cast<ScalarClusterable*>(clusters[1])->Mean();
      mean = (mean1 + mean2) / 2;
    }
    output.Stream() << mean;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
