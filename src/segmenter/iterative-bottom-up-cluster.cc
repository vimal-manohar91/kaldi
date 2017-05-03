// segmenter/iterative-bottom-up-cluster.cc

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

#include "segmenter/iterative-bottom-up-cluster.h"
#include "segmenter/plda-clusterable.h"
#include "segmenter/group-clusterable.h"
#include "segmenter/ivector-clusterable.h"
#include "tree/cluster-utils.h"

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

template<class T> 
class IterativeBottomUpClusterer {
 public:
  IterativeBottomUpClusterer(
    const IterativeBottomUpClusteringOptions &opts,
    BaseFloat max_merge_threshold, int32 min_clusters, 
    const std::vector<Clusterable*> &points,
    std::vector<Clusterable*> *clusters_out,
    std::vector<int32> *assignments_out)
    : opts_(opts), max_merge_thresh_(max_merge_threshold),
    min_clust_(min_clusters),
    points_(points), clusters_out_(clusters_out),
    assignments_out_(assignments_out) { 
      KALDI_ASSERT(points.size() == 0 || 
                   dynamic_cast<T*>(points[0]));
    }
  void Cluster();

 private:
  void SplitClusterToPoints(
    Clusterable *clusterable,
    std::vector<Clusterable *> *split_clusterables);

  void ClusterOneIter(
      int32 num_clusters_stage1, int32 num_clusters_final,
      BaseFloat max_merge_threshold,
      std::vector<std::vector<Clusterable*> > *clusterables_out,
      std::vector<std::vector<int32> > *assignments_out);
  
  const IterativeBottomUpClusteringOptions &opts_;
  BaseFloat max_merge_thresh_;
  int32 min_clust_;
  const std::vector<Clusterable*> &points_;
  std::vector<Clusterable*> *clusters_out_;
  std::vector<int32> *assignments_out_;
  
  std::vector<std::vector<Clusterable*> > clusterables_;
};

template<class T>
void IterativeBottomUpClusterer<T>::Cluster() {
  if (opts_.num_clusters_intermediate == 0 || opts_.num_iters <= 1) {
    ClusterBottomUp(points_, max_merge_thresh_, 
                    min_clust_, NULL,
                    assignments_out_);
    return;
  }

  assignments_out_->clear();
  assignments_out_->resize(points_.size());
  //(points_.size() + compartment_size_ - 1) / compartment_size_;
  clusterables_.resize(opts_.num_clusters_intermediate);

  int32 compartment_size = points_.size() / opts_.num_clusters_intermediate;
  
  if (opts_.num_clusters_intermediate * compartment_size < points_.size()) {
    compartment_size++;
  }

  for (size_t i = 0; i < points_.size(); i++) {
    int32 compartment = i / compartment_size;
    (*assignments_out_)[i] = compartment;

    T *t = dynamic_cast<T*>(points_[i]);
    KALDI_ASSERT(t != NULL);
    clusterables_[compartment].push_back(t->Copy());
  }

  for (int32 iter = 0; iter < opts_.num_iters; iter++) {
    std::vector<std::vector<Clusterable *> > clusterables_out;
    std::vector<std::vector<int32> > assignments_out;
    
    int32 num_compartments = clusterables_.size();
    ClusterOneIter(
        opts_.num_clusters_intermediate,
        iter < opts_.num_iters - 1 ? num_compartments : min_clust_,
        max_merge_thresh_, 
        iter < opts_.num_iters ? &clusterables_out : NULL,
        &assignments_out);

    for (int32 c = 0; c < num_compartments; c++ ) {
      for (int32 i = 0; i < assignments_out[c].size(); i++) {

        if (T *pc = dynamic_cast<T*>(clusterables_[c][i])) {
          const std::set<int32> &this_points = pc->points();

          KALDI_ASSERT(this_points.size() == 1);
          (*assignments_out_)[*(this_points.begin())] = 
            assignments_out[c][i];
        } else {
          KALDI_ERR << "Clusterable object must be a PointsClusterable object to "
                    << "use IterativeBottomUpClusterer.";
        }
      }
    }
    
    for (int32 c = 0; c < num_compartments; c++) {
      DeletePointers(&(clusterables_[c]));
    }

    if (iter < opts_.num_iters - 1) {
      clusterables_.clear();
      clusterables_ = clusterables_out;
    }
  }
}

template <class T> 
void IterativeBottomUpClusterer<T>::SplitClusterToPoints(
    Clusterable *clusterable,
    std::vector<Clusterable *> *split_clusterables) {
  KALDI_ASSERT(dynamic_cast<T*> (clusterable) != NULL);
  KALDI_ASSERT(split_clusterables);

  if (T *pc = dynamic_cast<T*> (clusterable)) {
    const std::set<int32> &this_points = pc->points();

    for (std::set<int32>::const_iterator it = this_points.begin();
         it != this_points.end(); ++it) {
      T *t = static_cast<T*> (points_[*it]);
      KALDI_ASSERT(t != NULL);
      split_clusterables->push_back(t->Copy());
    }
  } else {
    KALDI_ERR << "Clusterable object must be a PointsClusterable object to "
              << "use IterativeBottomUpClusterer.";
  }
}

template <class T>
void IterativeBottomUpClusterer<T>::ClusterOneIter(
    int32 num_clusters_stage1, int32 num_clusters_final, 
    BaseFloat max_merge_threshold,
    std::vector<std::vector<Clusterable *> > *clusterables_out,
    std::vector<std::vector<int32> > *assignments_out) {
  int32 num_compartments = clusterables_.size();

  std::vector<std::vector<Clusterable *> > clusters_stage1;
  std::vector<std::vector<int32> > assignments_stage1;

  ClusterBottomUpCompartmentalized(
      clusterables_, max_merge_threshold, 
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

void CompartmentalizeAndClusterBottomUpPlda(
    const IterativeBottomUpClusteringOptions &opts,
    BaseFloat max_merge_thresh, int32 min_clust,
    const std::vector<Clusterable*> &points,
    std::vector<Clusterable*> *clusters_out,
    std::vector<int32> *assignments_out) {
  IterativeBottomUpClusterer<PldaClusterable> clusterer(
      opts, max_merge_thresh, min_clust, points, 
      clusters_out, assignments_out);
  clusterer.Cluster();
}

void CompartmentalizeAndClusterBottomUpGroup(
    const IterativeBottomUpClusteringOptions &opts,
    BaseFloat max_merge_thresh, int32 min_clust,
    const std::vector<Clusterable*> &points,
    std::vector<Clusterable*> *clusters_out,
    std::vector<int32> *assignments_out) {
  IterativeBottomUpClusterer<GroupClusterable> clusterer(
      opts, max_merge_thresh, min_clust, points, 
      clusters_out, assignments_out);
  clusterer.Cluster();
}

void CompartmentalizeAndClusterBottomUpIvector(
    const IterativeBottomUpClusteringOptions &opts,
    BaseFloat max_merge_thresh, int32 min_clust,
    const std::vector<Clusterable*> &points,
    std::vector<Clusterable*> *clusters_out,
    std::vector<int32> *assignments_out) {
  IterativeBottomUpClusterer<IvectorClusterable> clusterer(
      opts, max_merge_thresh, min_clust, points, 
      clusters_out, assignments_out);
  clusterer.Cluster();
}

}  // end namespace kaldi
