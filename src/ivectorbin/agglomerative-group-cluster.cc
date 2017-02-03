// ivectorbin/agglomerative-group-cluster.cc

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
#include "ivector/group-clusterable.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Cluster ivectors using cosine distance\n"
      "TODO better documentation\n"
      "Usage: agglomerative-cluster-ivectors [options] <ivectors-rspecifier> "
      "<spk2utt-rspecifier> <labels-wspecifier>\n"
      "e.g.: \n"
      " agglomerative-cluster-ivectors ark:ivectors.ark ark:spk2utt \n"
      "   ark,t:labels.txt\n";

    ParseOptions po(usage);
    std::string utt2num_rspecifier, utt2num_frames_rspecifier;
    BaseFloat threshold = 0;
    int32 compartment_size = 256;

    po.Register("compartment-size", &compartment_size, 
                "First cluster within compartments of this size.");
    po.Register("utt2num-rspecifier", &utt2num_rspecifier,
      "If supplied, clustering creates exactly this many clusters for each"
      "utterance and the option --threshold is ignored.");
    po.Register("utt2num-frames-rspecifier", &utt2num_frames_rspecifier,
      "The number of frames in each utterance.");
    po.Register("threshold", &threshold, "Merging clusters if their distance"
      "is less than this threshold.");

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
    RandomAccessInt32Reader utt2num_reader(utt2num_rspecifier);
    Int32Writer label_writer(label_wspecifier);

    int32 num_err = 0, num_done = 0;
    for (; !scores_reader.Done(); scores_reader.Next()) {
      const std::string &spk = scores_reader.Key();
      Matrix<BaseFloat> scores(scores_reader.Value());

      // Convert scores into distances.
      scores.Scale(-1.0);
      scores.Sigmoid(scores);

      if (!spk2utt_reader.HasKey(spk)) {
        KALDI_WARN << "Could not find uttlist for speaker " << spk
                   << " in " << spk2utt_rspecifier;
        num_err++;
        continue;
      }

      const std::vector<std::string> &uttlist = spk2utt_reader.Value(spk);

      std::vector<std::vector<Clusterable *> > compartmentalized_clusters;
      std::vector<std::vector<int32> > compartmentalized_spk_ids;

      int32 this_num_utts = uttlist.size();
      int32 num_compartments = (this_num_utts + compartment_size - 1)
                                / compartment_size;
      std::vector<std::vector<std::string> > compartmentalized_utts(
          num_compartments, std::vector<std::string>());
      std::vector<std::vector<Clusterable*> > clusterables(
          num_compartments, std::vector<Clusterable*>());
      
      {
        for (size_t i = 0; i < this_num_utts; i++) {
          int32 compartment = i / compartment_size;
          std::set<int32> points;
          points.insert(i);

          clusterables[compartment].push_back(
              new GroupClusterable(points, &scores));
          compartmentalized_utts[compartment].push_back(uttlist[i]);
        }
        
        if (utt2num_rspecifier.size()) {
          int32 num_speakers = utt2num_reader.Value(spk);
          ClusterBottomUpCompartmentalized(
              clusterables, std::numeric_limits<BaseFloat>::max(),
              std::max(num_compartments, num_speakers), 
              &compartmentalized_clusters, 
              &compartmentalized_spk_ids);
        } else {
          ClusterBottomUpCompartmentalized(clusterables, 
              1.0 / (1 + Exp(-threshold)), 
              std::max(num_compartments, 1), 
              &compartmentalized_clusters, 
              &compartmentalized_spk_ids);
        }
      }

      {
        std::vector<Clusterable*> clusterables_tmp;
        std::vector<std::vector<int32> > compartmentalized_spk2cluster(
            num_compartments, std::vector<int32>());
        std::vector<int32> spk_ids;
        for (int32 c = 0, id = 0; c < num_compartments; c++) {
          compartmentalized_spk2cluster[c].resize(
            compartmentalized_clusters[c].size()); 
          for (int32 i = 0; i < compartmentalized_clusters[c].size(); 
               i++, id++) {
            clusterables_tmp.push_back(compartmentalized_clusters[c][i]);
            compartmentalized_spk2cluster[c][i] = id;
          }
        }

        if (utt2num_rspecifier.size()) {
          int32 num_speakers = utt2num_reader.Value(spk);
          ClusterBottomUp(
              clusterables_tmp, std::numeric_limits<BaseFloat>::max(),
              num_speakers, NULL, &spk_ids);
        } else {
          ClusterBottomUp(clusterables_tmp, 
              1.0 / (1 + Exp(-threshold)), 1,
              NULL, &spk_ids);
        }
      
        for (int32 c = 0; c < num_compartments; c++) {
          KALDI_ASSERT (compartmentalized_utts[c].size() == 
                        compartmentalized_spk_ids[c].size());
          for (int32 i = 0; i < compartmentalized_utts[c].size(); i++) {
            int32 spk_in_compartment = compartmentalized_spk_ids[c][i];
            int32 tmp_id = 
              compartmentalized_spk2cluster[c][spk_in_compartment];
            label_writer.Write(compartmentalized_utts[c][i], spk_ids[tmp_id]);
          }
        }
      }

      for (int32 c = 0; c < num_compartments; c++) {
        DeletePointers(&(clusterables[c]));
      }
      num_done++;
    }

    return (num_done > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
