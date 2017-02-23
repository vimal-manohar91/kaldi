// segmenterbin/agglomerative-cluster-vector-adjacency.cc

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
#include "segmenter/pair-clusterable.h"
#include "segmenter/adjacency-clusterable.h"
#include "segmenter/segmentation.h"

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
    const Matrix<BaseFloat> &matrix, const Vector<BaseFloat> &row_weights,
    Clusterable *clusterable,
    std::vector<Clusterable *> *split_clusterables) {
  KALDI_ASSERT(clusterable->Type() == "pair");
  KALDI_ASSERT(split_clusterables);
  PairClusterable *pc = NULL;
  pc = static_cast<PairClusterable*>(clusterable);
  AdjacencyClusterable *ac = NULL;
  ac = static_cast<AdjacencyClusterable*>(pc->clusterable2());
  const std::set<int32> &points = ac->points();
            
  for (std::set<int32>::const_iterator it = points.begin();
       it != points.end(); ++it) {
    std::set<int32> pts;
    pts.insert(*it);
    split_clusterables->push_back(new PairClusterable(
          new VectorClusterable(Vector<BaseFloat>(matrix.Row(*it)), row_weights(*it)),
          new AdjacencyClusterable(pts, ac->start_times(), ac->end_times()),
          pc->Weight1(), pc->Weight2()));
  }
}

void ClusterOneIter(
    const Matrix<BaseFloat> &matrix, const Vector<BaseFloat> &row_weights,
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
      SplitClusterToPoints(matrix, row_weights,
                           clusters_stage2[c], &((*clusterables_out)[c]));
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
      "Cluster vectors using cosine distance\n"
      "TODO better documentation\n"
      "Usage: agglomerative-cluster-vector-adjacency [options] "
      "<reco2utt-rspecifier> <vector-rspecifier> <segmentation-rspecifier> "
      "<labels-wspecifier>\n"
      "e.g.: \n"
      " agglomerative-cluster-vector-adjacency ark:reco2utt \n"
      "   scp:ivectors.scp \"ark:segmentation-init-from-segments --shift-to-zero=false --frame-overlap=0.0 segments ark:- |\" ark,t:labels.txt\n";

    ParseOptions po(usage);
    std::string reco2num_spk_rspecifier, utt2num_frames_rspecifier;
    BaseFloat threshold = 400;
    int32 compartment_size = 0;
    int32 num_iters = 3;
    int32 num_clusters_intermediate = 256;
    BaseFloat adjacency_factor = 0.01;

    po.Register("compartment-size", &compartment_size, 
                "If specified, first cluster within compartments of this size.");
    po.Register("reco2num-spk-rspecifier", &reco2num_spk_rspecifier,
      "If supplied, clustering creates exactly this many clusters for each"
      "utterance and the option --threshold is ignored.");
    po.Register("utt2num-frames-rspecifier", &utt2num_frames_rspecifier,
                "The number of frames in each utterance.");
    po.Register("threshold", &threshold, 
                "Merging clusters if their distance"
                "is less than this threshold.");
    po.Register("num-iters", &num_iters, "Number of iterations of clustering");
    po.Register("num-clusters-intermediate", &num_clusters_intermediate,
                "Cluster first into this many clusters using "
                "compartmentalized bottom-up clustering.");
    po.Register("adjacency-factor", &adjacency_factor, 
                "Scale of adjacency penalty in the objective function.");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string reco2utt_rspecifier = po.GetArg(1),
      vector_rspecifier = po.GetArg(2),
      segmentation_rspecifier = po.GetArg(3),
      label_wspecifier = po.GetArg(4);

    // TODO  Maybe should make the PLDA scoring binary output segmentation so that this can read it
    // directly. If not, at least make sure the utt2seg in that binary is NOT sorted. Might sort it in a different
    // order than here.
    SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
    RandomAccessBaseFloatVectorReader vector_reader(vector_rspecifier);
    segmenter::RandomAccessSegmentationReader segmentation_reader(
        segmentation_rspecifier);
    RandomAccessInt32Reader reco2num_spk_reader(reco2num_spk_rspecifier);
    RandomAccessInt32Reader utt2num_frames_reader(utt2num_frames_rspecifier);
    Int32Writer label_writer(label_wspecifier);

    int32 num_err = 0, num_done = 0;
    for (; !reco2utt_reader.Done(); reco2utt_reader.Next()) {
      const std::string &reco = reco2utt_reader.Key();
      const std::vector<std::string> &uttlist = reco2utt_reader.Value();

      int32 this_num_utts = uttlist.size();
      std::vector<int32> utt2compartment(this_num_utts);
        
      std::vector<std::vector<Clusterable*> > clusterables;
      std::vector<Clusterable*> clusterables_simple;

      Vector<BaseFloat> start_times(this_num_utts);
      Vector<BaseFloat> end_times(this_num_utts);

      Matrix<BaseFloat> matrix;
      Vector<BaseFloat> row_weights;

      std::vector<std::pair<Vector<BaseFloat>, int32> > vectors(this_num_utts);


      if (compartment_size > 0) {
        int32 num_compartments = 
          (this_num_utts + compartment_size - 1) / compartment_size;
        clusterables.resize(num_compartments);
      }

      for (size_t i = 0; i < this_num_utts; i++) {
        utt2compartment[i] = compartment_size > 0 ? i / compartment_size : i;

        const std::string &utt = uttlist[i];

        if (!vector_reader.HasKey(utt)) {
          KALDI_WARN << "Could not find Vector for utterance " << utt
                     << " in archive " << vector_rspecifier << "; "
                     << "skipping utterance.";
          num_err++;
          continue;
        }
        
        if (!segmentation_reader.HasKey(utt)) {
          KALDI_WARN << "Could not find start and end frames for "
                     << "utterance " << utt
                     << " in archive " << segmentation_rspecifier << "; "
                     << "skipping utterance.";
          num_err++;
          continue;
        }

        int32 num_frames = 1;
        if (!utt2num_frames_rspecifier.empty()) {
          if (!utt2num_frames_reader.HasKey(utt)) {
            KALDI_WARN << "Could not read num-frames for "
                       << "utterance " << utt
                       << " in archive " << utt2num_frames_rspecifier << "; "
                       << "skipping utterance.";
            num_err++;
            continue;
          }
          num_frames = utt2num_frames_reader.Value(utt);
        }

        std::set<int32> points;
        points.insert(i);

        const Vector<BaseFloat> &vector = vector_reader.Value(utt);
        const segmenter::Segmentation &seg = segmentation_reader.Value(utt);
        
        if (matrix.NumRows() == 0) {
          matrix.Resize(uttlist.size(), vector.Dim());
          row_weights.Resize(uttlist.size());
        }

        matrix.CopyRowFromVec(vector, i);
        row_weights(i) = num_frames;

        if (seg.Dim() != 1) {
          KALDI_ERR << "segmentation is not kaldi segments converted to "
                    << "Segmentation format.";
        }

        start_times(i) = seg.Begin()->start_frame;
        end_times(i) = seg.Begin()->end_frame;

        if (compartment_size > 0) {
          int32 compartment = i / compartment_size;
          clusterables[compartment].push_back(new PairClusterable(
                new VectorClusterable(vector, num_frames),
                new AdjacencyClusterable(points, &start_times, &end_times),
                1.0, -adjacency_factor));
        } else {
          clusterables_simple.push_back(new PairClusterable(
              new VectorClusterable(vector, num_frames),
              new AdjacencyClusterable(points, &start_times, &end_times),
              1.0, -adjacency_factor));
        }
      }

      int32 this_num_speakers = 1;
      if (!reco2num_spk_rspecifier.empty()) {
        this_num_speakers = reco2num_spk_reader.Value(reco);
      } 

      if (compartment_size > 0) {
        for (int32 iter = 0; iter < num_iters; iter++) {
          std::vector<std::vector<Clusterable *> > clusterables_out;
          std::vector<std::vector<int32> > assignments_out;

          int32 num_compartments = clusterables.size();
          ClusterOneIter(
              matrix, row_weights,
              clusterables, num_clusters_intermediate,
              iter < num_iters - 1 ? num_compartments : this_num_speakers,
              (threshold < 0 || !reco2num_spk_rspecifier.empty()) 
              ? std::numeric_limits<BaseFloat>::max() : threshold, 
              &assignments_out,
              iter < num_iters ? &clusterables_out : NULL);

          for (int32 c = 0; c < num_compartments; c++ ) {
            for (int32 i = 0; i < assignments_out[c].size(); i++) {
              PairClusterable *pc = NULL;
              pc = static_cast<PairClusterable*>(clusterables[c][i]);
              AdjacencyClusterable *ac = NULL;
              ac = static_cast<AdjacencyClusterable*>(pc->clusterable2());
              const std::set<int32> &points = ac->points();
              
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
      } else {
        ClusterBottomUp(clusterables_simple, 
                        (threshold < 0 || !reco2num_spk_rspecifier.empty()) 
                        ? std::numeric_limits<BaseFloat>::max() : threshold, 
                        this_num_speakers,
                        NULL, &utt2compartment);
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
