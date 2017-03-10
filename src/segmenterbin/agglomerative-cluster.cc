// ivectorbin/agglomerative-cluster.cc

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
#include "tree/cluster-utils.h"
#include "tree/clusterable-classes.h"
#include "segmenter/group-clusterable.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Cluster matrices of scores per utterance. Used in diarization\n"
      "TODO better documentation\n"
      "Usage: agglomerative-cluster [options] <scores-rspecifier> "
      "<spk2utt-rspecifier> <labels-wspecifier>\n"
      "e.g.: \n"
      " agglomerative-cluster ark:scores.ark ark:spk2utt \n"
      "   ark,t:labels.txt\n";

    ParseOptions po(usage);
    std::string utt2num_spk_rspecifier, utt2num_frames_rspecifier;
    BaseFloat threshold = 0;
    bool apply_sigmoid = true;

    po.Register("utt2num-spk-rspecifier", &utt2num_spk_rspecifier,
      "If supplied, clustering creates exactly this many clusters for each"
      "utterance and the option --threshold is ignored.");
    po.Register("utt2num-frames-rspecifier", &utt2num_frames_rspecifier,
      "The number of frames in each utterance.");
    po.Register("threshold", &threshold, "Merging clusters if their distance"
                "is less than this threshold.");
    po.Register("apply-sigmoid", &apply_sigmoid, "Apply sigmoid transformation "
        "distances");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string scores_rspecifier = po.GetArg(1),
      spk2utt_rspecifier = po.GetArg(2),
      label_wspecifier = po.GetArg(3);

    // TODO  Maybe should make the PLDA scoring binary output segmentation so that this can read it
    // directly. If not, at least make sure the utt2seg in that binary is NOT sorted. Might sort it in a different
    // order than here.
    SequentialBaseFloatMatrixReader scores_reader(scores_rspecifier);
    RandomAccessTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessInt32Reader utt2num_reader(utt2num_spk_rspecifier);
    Int32Writer label_writer(label_wspecifier);

    int32 num_err = 0, num_done = 0;
    for (; !scores_reader.Done(); scores_reader.Next()) {
      const std::string &spk = scores_reader.Key();
      Matrix<BaseFloat> scores(scores_reader.Value());

      // Convert scores into distances.
      scores.Scale(-1.0);
      
      if (apply_sigmoid)
        scores.Sigmoid(scores);

      if (!spk2utt_reader.HasKey(spk)) {
        KALDI_WARN << "Could not find uttlist for speaker " << spk
                   << " in " << spk2utt_rspecifier;
        num_err++;
        continue;
      }

      const std::vector<std::string> &uttlist = spk2utt_reader.Value(spk);

      std::vector<Clusterable*> clusterables;
      std::vector<int32> spk_ids;

      int32 this_num_utts = uttlist.size();

      for (size_t i = 0; i < this_num_utts; i++) {
        std::set<int32> points;
        points.insert(i);
        clusterables.push_back(new GroupClusterable(points, &scores));
      }

      if (!utt2num_spk_rspecifier.empty()) {
        int32 num_speakers = utt2num_reader.Value(spk);
        ClusterBottomUp(clusterables, std::numeric_limits<BaseFloat>::max(),
          num_speakers, NULL, &spk_ids);
      } else {
        ClusterBottomUp(clusterables, 
            apply_sigmoid ? 1.0 / (1 + Exp(-threshold)) : threshold,
            1, NULL, &spk_ids);
      }

      for (size_t i = 0; i < this_num_utts; i++) {
        label_writer.Write(uttlist[i], spk_ids[i]);
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
