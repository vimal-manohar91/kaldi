// ivectorbin/agglomerative-group-cluster.cc

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
#include "util/stl-utils.h"
#include "tree/cluster-utils.h"
#include "tree/clusterable-classes.h"
#include "ivector/group-clusterable.h"

namespace kaldi {

void FlattenCompartments(
    const std::vector<std::vector<Clusterable *> > 
      &compartmentalized_clusters,
    std::vector<Clusterable *> *clusterables, 
    std::vector<std::vector<int32> > *compartmentalized_assignment2id) {
  
  int32 num_compartments = compartmentalized_clusters.size();
  compartmentalized_assignment2id->resize(num_compartments);
  for (int32 c = 0, id = 0; c < num_compartments; c++) {
    (*compartmentalized_assignment2id)[c].resize(
        compartmentalized_clusters[c].size());
    for (int32 i = 0; i < compartmentalized_clusters[c].size(); 
         i++, id++) {
      clusterables->push_back(
          compartmentalized_clusters[c][i]);
      (*compartmentalized_assignment2id)[c][i] = id;
    }
  }
}

void SplitClusterToPoints(
    Clusterable *clusterable,
    std::vector<Clusterable *> *split_clusterables) {
  KALDI_ASSERT(clusterable->Type() == "group");
  KALDI_ASSERT(split_clusterables);
  GroupClusterable *gc = NULL;

  gc = static_cast<GroupClusterable *>(clusterable);

  const std::set<int32> &points = gc->points();

  for (std::set<int32>::const_iterator it = points.begin();
       it != points.end(); ++it) {
    std::set<int32> pts;
    pts.insert(*it);
    split_clusterables->push_back(new GroupClusterable(pts, gc->scores()));
  }
}

void ClusterOneIter(
    const std::vector<std::vector<Clusterable *> > &clusterables,
    int32 num_clusters_stage1, int32 num_clusters_final, 
    BaseFloat max_merge_threshold,
    std::vector<std::vector<int32> > *assignments_out,
    std::vector<std::vector<Clusterable *> > *clusterables_out = NULL) {
  int32 num_compartments = clusterables.size();

  std::vector<std::vector<Clusterable *> > clusters_stage1;
  std::vector<std::vector<int32> > assignments_stage1;

  ClusterBottomUpCompartmentalized(
      clusterables, max_merge_threshold, 
      std::max(num_compartments, num_clusters_stage1),
      &clusters_stage1, &assignments_stage1);

  std::vector<Clusterable *> clusterables_stage1;
  std::vector<std::vector<int32> > stage1_assignment2id;
  FlattenCompartments(clusters_stage1, &clusterables_stage1, 
                      &stage1_assignment2id);

  std::vector<Clusterable *> clusters_stage2;
  std::vector<int32> assignments_stage2;
  ClusterBottomUp(clusterables_stage1, max_merge_threshold,
                  num_clusters_final,
                  (clusterables_out ? &clusters_stage2 : NULL),
                  &assignments_stage2);

  assignments_out->resize(num_compartments);
  for (int32 c = 0; c < num_compartments; c++) {
    (*assignments_out)[c].resize(assignments_stage1[c].size());
    for (int32 i = 0; i < assignments_stage1[c].size(); i++) {
      int32 stage1_assignment = assignments_stage1[c][i];
      int32 tmp_id = 
        stage1_assignment2id[c][stage1_assignment];
      (*assignments_out)[c][i] = assignments_stage2[tmp_id];
    }
  }
  
  if (clusterables_out) {
    clusterables_out->resize(clusters_stage2.size());
    for (int32 c = 0; c < clusters_stage2.size(); c++) {
      SplitClusterToPoints(clusters_stage2[c], &((*clusterables_out)[c]));
    }
  }
}

}  // end namespace kaldi

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Cluster ivectors using cosine distance\n"
      "TODO better documentation\n"
      "Usage: agglomerative-group-cluster [options] <scores-rspecifier> "
      "<reco2utt-rspecifier> <labels-wspecifier>\n"
      "e.g.: \n"
      " agglomerative-cluster-ivectors ark:ivectors.ark ark:spk2utt \n"
      "   ark,t:labels.txt\n";

    ParseOptions po(usage);
    std::string utt2num_spk_rspecifier, utt2num_frames_rspecifier;
    BaseFloat threshold = 0;
    int32 compartment_size = 256;
    int32 num_iters = 3;
    int32 num_clusters_intermediate = 256;

    po.Register("compartment-size", &compartment_size, 
                "First cluster within compartments of this size.");
    po.Register("utt2num-spk-rspecifier", &utt2num_spk_rspecifier,
      "If supplied, clustering creates exactly this many clusters for each"
      "utterance and the option --threshold is ignored.");
    po.Register("utt2num-frames-rspecifier", &utt2num_frames_rspecifier,
      "The number of frames in each utterance.");
    po.Register("threshold", &threshold, "Merging clusters if their distance"
      "is less than this threshold.");
    po.Register("num-iters", &num_iters, "Number of iterations of clustering");
    po.Register("num-clusters-intermediate", &num_clusters_intermediate,
                "Cluster first into this many clusters using "
                "compartmentalized bottom-up clustering.");

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
    RandomAccessInt32Reader utt2num_spk_reader(utt2num_spk_rspecifier);
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

      int32 this_num_utts = uttlist.size();
      std::vector<int32> utt2compartment(this_num_utts);
      int32 num_compartments = 
        (this_num_utts + compartment_size - 1) / compartment_size;
        
      std::vector<std::vector<Clusterable *> > clusterables(num_compartments);
      for (size_t i = 0; i < this_num_utts; i++) {
        int32 compartment = i / compartment_size;
        utt2compartment[i] = compartment;

        std::set<int32> points;
        points.insert(i);

        clusterables[compartment].push_back(
            new GroupClusterable(points, &scores));
      }
        
      int32 this_num_speakers = 1;
      if (utt2num_spk_rspecifier.size()) {
        this_num_speakers = utt2num_spk_reader.Value(spk);
      } 

      for (int32 iter = 0; iter < num_iters; iter++) {
        std::vector<std::vector<Clusterable *> > clusterables_out;
        std::vector<std::vector<int32> > assignments_out;

        ClusterOneIter(
            clusterables, num_clusters_intermediate,
            iter < num_iters - 1 ? num_compartments : this_num_speakers,
            1.0 / (1 + Exp(threshold)),
            &assignments_out,
            iter < num_iters ? &clusterables_out : NULL);

        for (int32 c = 0; c < num_compartments; c++ ) {
          for (int32 i = 0; i < assignments_out[c].size(); i++) {
            GroupClusterable *gc = NULL;
              
            gc = static_cast<GroupClusterable *>(clusterables[c][i]);
            const std::set<int32> &points = gc->points();

            KALDI_ASSERT(points.size() == 1);
            utt2compartment[*(points.begin())] = 
              assignments_out[c][i];
          }
        }

        for (int32 c = 0; c < num_compartments; c++) {
          DeletePointers(&(clusterables[c]));
        }
        
        if (iter < num_iters - 1) {
          clusterables.clear();
          clusterables = clusterables_out;
        }
      }

      for (size_t i = 0; i < this_num_utts; i++) {
        label_writer.Write(uttlist[i], utt2compartment[i]);
      }

      num_done++;
    }

    return (num_done > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
