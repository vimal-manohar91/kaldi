// segmenterbin/agglomerative-cluster.cc

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
#include "util/stl-utils.h"
#include "tree/cluster-utils.h"
#include "tree/clusterable-classes.h"
#include "segmenter/group-clusterable.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Cluster score matrix using average pair-wise distance\n"
      "TODO better documentation\n"
      "Usage: agglomerative-cluster [options] <scores-rspecifier> "
      "<reco2utt-rspecifier> <labels-wspecifier> [<utt2spk-wspecifier>]\n"
      "e.g.: \n"
      " agglomerative-cluster ark:scores.ark ark:reco2utt \n"
      "   ark,t:labels.txt\n";

    ParseOptions po(usage);
    std::string reco2num_spk_rspecifier, utt2num_frames_rspecifier;
    std::string thresholds_rspecifier;
    BaseFloat threshold = 0;
    bool apply_sigmoid = true;

    po.Register("reco2num-spk-rspecifier", &reco2num_spk_rspecifier,
                "If supplied, clustering creates exactly this many clusters "
                "for each recording and the option --threshold is ignored.");
    po.Register("utt2num-frames-rspecifier", &utt2num_frames_rspecifier,
                "The number of frames in each utterance.");
    po.Register("threshold", &threshold, 
                "Merging clusters if their distance"
                "is less than this threshold.");
    po.Register("thresholds-rspecifier", &thresholds_rspecifier,
                "If specified, applies a per-recording threshold; "
                "overrides --threshold.");
    po.Register("apply-sigmoid", &apply_sigmoid, "Apply sigmoid transformation "
        "distances");

    GroupClusterableOptions opts;
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3 && po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string scores_rspecifier = po.GetArg(1),
      reco2utt_rspecifier = po.GetArg(2),
      label_wspecifier = po.GetArg(3),
      utt2spk_wspecifier = po.GetOptArg(4);

    SequentialBaseFloatMatrixReader scores_reader(scores_rspecifier);
    RandomAccessTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
    RandomAccessInt32Reader reco2num_spk_reader(reco2num_spk_rspecifier);
    RandomAccessInt32Reader utt2num_frames_reader(utt2num_frames_rspecifier);
    RandomAccessBaseFloatReader thresholds_reader(thresholds_rspecifier);
    Int32Writer label_writer(label_wspecifier);
    TokenWriter utt2spk_writer(utt2spk_wspecifier);

    int32 num_err = 0, num_done = 0;
    for (; !scores_reader.Done(); scores_reader.Next()) {
      const std::string &reco = scores_reader.Key();
      Matrix<BaseFloat> scores(scores_reader.Value());

      // Convert scores into distances.
      scores.Scale(-1.0);

      if (apply_sigmoid)
        scores.Sigmoid(scores);

      if (!reco2utt_reader.HasKey(reco)) {
        KALDI_WARN << "Could not find uttlist for recording " << reco
                   << " in " << reco2utt_rspecifier;
        num_err++;
        continue;
      }

      const std::vector<std::string> &uttlist = reco2utt_reader.Value(reco);

      std::vector<Clusterable*> clusterables;

      for (size_t i = 0; i < uttlist.size(); i++) {
        std::set<int32> points;
        points.insert(i);

        clusterables.push_back(new GroupClusterable(opts, points, &scores));
      }

      BaseFloat this_threshold = threshold;
      if (!thresholds_rspecifier.empty()) {
        if (!thresholds_reader.HasKey(reco)) {
          KALDI_WARN << "Could not find threshold for recording " << reco 
                     << " in " << thresholds_rspecifier << "; using "
                     << "--threshold=" << threshold;
        } else {
          this_threshold = thresholds_reader.Value(reco);
        }
      } 

      int32 this_num_speakers = 1;
      std::vector<int32> utt2cluster(uttlist.size());
      if (!reco2num_spk_rspecifier.empty()) {
        if (!reco2num_spk_reader.HasKey(reco)) {
          KALDI_WARN << "Could not find num-speakers for recording "
                     << reco;
          num_err++;
          continue;
        }
        this_num_speakers = reco2num_spk_reader.Value(reco);
        ClusterBottomUp(clusterables, std::numeric_limits<BaseFloat>::max(),
          this_num_speakers, NULL, &utt2cluster);
      } else {
        ClusterBottomUp(clusterables, 
            apply_sigmoid ? 1.0 / (1 + Exp(-this_threshold)) : this_threshold,
            1, NULL, &utt2cluster);
      }

      for (size_t i = 0; i < uttlist.size(); i++) {
        label_writer.Write(uttlist[i], utt2cluster[i]);
      }

      if (!utt2spk_wspecifier.empty()) {
        for (size_t i = 0; i < uttlist.size(); i++) {
          std::ostringstream oss;
          oss << reco << "-" << utt2cluster[i];
          utt2spk_writer.Write(uttlist[i], oss.str());
        }
      }
      DeletePointers(&clusterables);
      num_done++;
    }

    return (num_done > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
