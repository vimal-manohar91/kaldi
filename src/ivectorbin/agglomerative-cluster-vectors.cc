// ivectorbin/agglomerative-cluster-vectors.cc

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
#include "ivector/ivector-clusterable.h"

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
    BaseFloat threshold = 0.5;
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

    std::string ivectors_rspecifier = po.GetArg(1),
      spk2utt_rspecifier = po.GetArg(2),
      label_wspecifier = po.GetArg(3);

    // TODO  Maybe should make the PLDA scoring binary output segmentation so that this can read it
    // directly. If not, at least make sure the utt2seg in that binary is NOT sorted. Might sort it in a different
    // order than here.
    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessBaseFloatVectorReader ivector_reader(ivectors_rspecifier);
    RandomAccessInt32Reader utt2num_reader(utt2num_rspecifier);
    RandomAccessInt32Reader utt2num_frames_reader(utt2num_frames_rspecifier);
    Int32Writer label_writer(label_wspecifier);

    int32 num_err = 0, num_done = 0;
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      const std::string &spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();

      std::vector<std::vector<Clusterable *> > compartmentalized_clusters;
      std::vector<std::vector<int32> > compartmentalized_spk_ids;

      int32 this_num_utts = uttlist.size();
      int32 num_compartments = (this_num_utts + compartment_size - 1)
                                / compartment_size;
      std::vector<std::vector<std::string> > compartmentalized_utts(
          num_compartments, std::vector<std::string>());
      
      {
        std::vector<std::vector<Clusterable*> > clusterables(
            num_compartments, std::vector<Clusterable*>());

        for (size_t i = 0; i < this_num_utts; i++) {
          if (!ivector_reader.HasKey(uttlist[i])) {
            KALDI_WARN << "Could not find ivector for key " << uttlist[i]
                       << " in " << ivectors_rspecifier;
            num_err++;
          }

          BaseFloat weight = 1.0;
          if (!utt2num_frames_rspecifier.empty()) {
            if (!utt2num_frames_reader.HasKey(uttlist[i])) {
              KALDI_WARN << "Could not find counts for key " << uttlist[i]
                         << " in " << utt2num_frames_rspecifier;
              num_err++;
              continue;
            }
            weight = utt2num_frames_reader.Value(uttlist[i]);
          }

          int32 compartment = i / compartment_size;

          clusterables[compartment].push_back(
              new IvectorClusterable(ivector_reader.Value(uttlist[i]), weight));
          compartmentalized_utts[compartment].push_back(uttlist[i]);
          num_done++;
        }
        
        if (utt2num_rspecifier.size()) {
          int32 num_speakers = utt2num_reader.Value(spk);
          ClusterBottomUpCompartmentalized(
              clusterables, std::numeric_limits<BaseFloat>::max(),
              std::max(num_speakers, num_compartments),
              &compartmentalized_clusters, 
              &compartmentalized_spk_ids);
        } else {
          ClusterBottomUpCompartmentalized(clusterables, threshold, 
              std::max(1, num_compartments), 
              &compartmentalized_clusters, 
              &compartmentalized_spk_ids);
        }
 
        for (int32 c = 0; c < num_compartments; c++) {
          DeletePointers(&(clusterables[c]));
        }
      }

      {
        std::vector<Clusterable*> clusterables;
        std::vector<int32> spk_ids;
        for (int32 c = 0; c < num_compartments; c++) {
          for (int32 i = 0; i < compartmentalized_clusters[c].size(); i++) {
            clusterables.push_back(compartmentalized_clusters[c][i]);
          }
        }

        if (utt2num_rspecifier.size()) {
          int32 num_speakers = utt2num_reader.Value(spk);
          ClusterBottomUp(
              clusterables, std::numeric_limits<BaseFloat>::max(),
              num_speakers, NULL, &spk_ids);
        } else {
          ClusterBottomUp(clusterables, threshold, 1, 
              NULL, &spk_ids);
        }
      
        DeletePointers(&clusterables);

        int32 id = 0;
        for (int32 c = 0; c < num_compartments; c++) {
          for (int32 i = 0; i < compartmentalized_clusters[c].size(); 
               i++, id++) {
            label_writer.Write(compartmentalized_utts[c][i], spk_ids[id]);
          }
        }
      }
    }

    return (num_done > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

