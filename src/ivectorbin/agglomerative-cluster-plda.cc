// ivectorbin/agglomerative-cluster-plda.cc

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
#include "ivector/plda-clusterable.h"
#include "ivector/transform.h"

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
    const Matrix<BaseFloat> &ivector_mat, const Vector<BaseFloat> &row_weights,
    Clusterable *clusterable,
    std::vector<Clusterable *> *split_clusterables) {
  KALDI_ASSERT(clusterable->Type() == "plda");
  KALDI_ASSERT(split_clusterables);
  PldaClusterable *pc = NULL;
  pc = static_cast<PldaClusterable *>(clusterable);
  const std::set<int32> &points = pc->points();

  for (std::set<int32>::const_iterator it = points.begin();
       it != points.end(); ++it) {
    std::set<int32> pts;
    pts.insert(*it);
    split_clusterables->push_back(new PldaClusterable(
          pc->opts_, pc->plda(), pts,
          Vector<BaseFloat>(ivector_mat.Row(*it)), row_weights(*it)));
  }
}

void ClusterOneIter(
    const Matrix<BaseFloat> &ivector_mat, const Vector<BaseFloat> &row_weights,
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
      SplitClusterToPoints(ivector_mat, row_weights,
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
      "Cluster ivectors using PLDA distance\n"
      "TODO better documentation\n"
      "Usage: agglomerative-cluster-plda [options] <plda> "
      "<reco2utt-rspecifier> <ivector-rspecifier> "
      "<labels-wspecifier>\n"
      "e.g.: \n"
      " agglomerative-cluster-plda plda ark:reco2utt scp:ivectors.scp \n"
      "   ark,t:labels.txt\n";

    ParseOptions po(usage);
    BaseFloat target_energy = 0.5;
    std::string reco2num_spk_rspecifier, utt2num_frames_rspecifier;
    BaseFloat threshold = 0;
    int32 compartment_size = 0;
    int32 num_iters = 3;
    int32 num_clusters_intermediate = 256;

    po.Register("target-energy", &target_energy,
                "Reduce dimensionality of i-vectors using PCA such "
                "that this fraction of the total energy remains.");
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
    
    PldaConfig plda_config;
    plda_config.Register(&po);

    PldaClusterableOptions plda_clusterable_opts;
    plda_clusterable_opts.Register(&po);

    po.Read(argc, argv);
    KALDI_ASSERT(target_energy <= 1.0);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda_rxfilename = po.GetArg(1),
      reco2utt_rspecifier = po.GetArg(2),
      ivector_rspecifier = po.GetArg(3),
      label_wspecifier = po.GetArg(4);

    Plda plda;
    ReadKaldiObject(plda_rxfilename, &plda);

    SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
    RandomAccessBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    Int32Writer label_writer(label_wspecifier);

    RandomAccessInt32Reader reco2num_spk_reader(reco2num_spk_rspecifier);
    RandomAccessInt32Reader utt2num_frames_reader(utt2num_frames_rspecifier);

    int32 num_err = 0, num_done = 0;
    for (; !reco2utt_reader.Done(); reco2utt_reader.Next()) {
      const std::string &reco = reco2utt_reader.Key();
      Plda this_plda(plda);

      const std::vector<std::string> &uttlist = reco2utt_reader.Value();
      
      std::vector<std::string> out_uttlist;

      Matrix<BaseFloat> ivector_mat;
      Vector<BaseFloat> row_weights(uttlist.size());
      
      int32 index = 0;
      for (size_t i = 0; i < uttlist.size(); i++) {
        const std::string &utt = uttlist[i];
        
        if (!ivector_reader.HasKey(utt)) {
          KALDI_ERR << "No iVector for utterance " << utt
                    << " in archive " << ivector_rspecifier;
          num_err++;
          continue;
        }
        
        int32 weight = 1;
        if (!utt2num_frames_rspecifier.empty()) {
          if (!utt2num_frames_reader.HasKey(utt)) {
            KALDI_ERR << "No weights for utterance " << utt << " in archive " 
                      << utt2num_frames_rspecifier;
            num_err++;
            continue;
          }
          weight = utt2num_frames_reader.Value(utt);
        }
        const Vector<BaseFloat> &ivector = ivector_reader.Value(utt);
        out_uttlist.push_back(utt);
        
        if (ivector_mat.NumRows() == 0)
          ivector_mat.Resize(uttlist.size(), ivector.Dim());
        ivector_mat.CopyRowFromVec(ivector, index);

        row_weights(index) = weight;
        index++;
      }

      ivector_mat.Resize(out_uttlist.size(), ivector_mat.NumCols(), kCopyData);
      row_weights.Resize(out_uttlist.size(), kCopyData);

      Matrix<BaseFloat> ivector_mat_pca, ivector_mat_plda, pca_transform;
      
      if (EstPca(ivector_mat, row_weights, target_energy, &pca_transform)) {
        // Apply PCA transform to the raw i-vectors.
        ApplyPca(ivector_mat, pca_transform, &ivector_mat_pca);

        // Apply PCA transform to the parameters of the PLDA model.
        this_plda.ApplyTransform(Matrix<double>(pca_transform));

        // Now transform the i-vectors using the reduced PLDA model.
        TransformIvectors(ivector_mat_pca, plda_config, this_plda,
                          &ivector_mat_plda);
      } else {
        TransformIvectors(ivector_mat, plda_config, this_plda,
                          &ivector_mat_plda);
      }

      int32 this_num_utts = out_uttlist.size();
      std::vector<int32> utt2compartment(this_num_utts);
        
      std::vector<std::vector<Clusterable*> > clusterables;
      std::vector<Clusterable*> clusterables_simple;

      if (compartment_size > 0) {
        int32 num_compartments = 
          (this_num_utts + compartment_size - 1) / compartment_size;
        clusterables.resize(num_compartments);
      }

      for (size_t i = 0; i < this_num_utts; i++) {
        utt2compartment[i] = compartment_size > 0 ? i / compartment_size : i;

        std::set<int32> points;
        points.insert(i);
        if (compartment_size > 0) {
          int32 compartment = i / compartment_size;
          clusterables[compartment].push_back(new PldaClusterable(
                plda_clusterable_opts,
                &this_plda, points, 
                Vector<BaseFloat>(ivector_mat_plda.Row(i)), row_weights(i)));
        } else {
          clusterables_simple.push_back(new PldaClusterable(
                plda_clusterable_opts,
                &this_plda, points, 
                Vector<BaseFloat>(ivector_mat_plda.Row(i)), row_weights(i)));
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
              ivector_mat_plda, row_weights,
              clusterables, num_clusters_intermediate,
              iter < num_iters - 1 ? num_compartments : this_num_speakers,
              (!reco2num_spk_rspecifier.empty()) 
              ? std::numeric_limits<BaseFloat>::max() : threshold, 
              &assignments_out,
              iter < num_iters ? &clusterables_out : NULL);

          for (int32 c = 0; c < num_compartments; c++ ) {
            for (int32 i = 0; i < assignments_out[c].size(); i++) {
              PldaClusterable *pc = NULL;
              pc = static_cast<PldaClusterable*>(clusterables[c][i]);
              const std::set<int32> &points = pc->points();
              
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
                        (!reco2num_spk_rspecifier.empty()) 
                        ? std::numeric_limits<BaseFloat>::max() : threshold, 
                        this_num_speakers,
                        NULL, &utt2compartment);
      }

      for (size_t i = 0; i < out_uttlist.size(); i++) {
        label_writer.Write(out_uttlist[i], utt2compartment[i]);
      }

      num_done++;
    } 

    return (num_done > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
