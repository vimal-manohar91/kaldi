// segmenter/iterative-bottom-up-cluster.h

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

#ifndef KALDI_SEGMENTER_ITERATIVE_BOTTOM_UP_CLUSTER_H_
#define KALDI_SEGMENTER_ITERATIVE_BOTTOM_UP_CLUSTER_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "itf/clusterable-itf.h"
#include "itf/options-itf.h"

namespace kaldi {

struct IterativeBottomUpClusteringOptions {
  int32 num_clusters_intermediate;
  int32 num_iters;
  int32 min_clusters;
  BaseFloat max_merge_threshold;

  IterativeBottomUpClusteringOptions()
    : num_clusters_intermediate(0), num_iters(0),
      min_clusters(1), 
      max_merge_threshold(std::numeric_limits<BaseFloat>::max()) { }

  void Register(OptionsItf *opts) {
    opts->Register("num-clusters-intermediate", &num_clusters_intermediate,
                   "Number of clusters in the intermediate stage.");
    opts->Register("num-iters", &num_iters,
                   "Number of iterations of bottom-up clustering.");
    opts->Register("min-clusters", &min_clusters,
                   "Stop at reaching this many clusters.");
    opts->Register("max-merge-threshold", &max_merge_threshold,
                   "Maximum distance allowed for merging clusters.");
  }
};

void CompartmentalizeAndClusterBottomUpPlda( 
    const IterativeBottomUpClusteringOptions &opts,
    BaseFloat max_merge_thresh, int32 min_clust,
    const std::vector<Clusterable*> &points,
    std::vector<Clusterable*> *clusters_out,
    std::vector<int32> *assignments_out);

void CompartmentalizeAndClusterBottomUpGroup( 
    const IterativeBottomUpClusteringOptions &opts,
    BaseFloat max_merge_thresh, int32 min_clust,
    const std::vector<Clusterable*> &points,
    std::vector<Clusterable*> *clusters_out,
    std::vector<int32> *assignments_out);

void CompartmentalizeAndClusterBottomUpIvector(
    const IterativeBottomUpClusteringOptions &opts,
    BaseFloat max_merge_thresh, int32 min_clust,
    const std::vector<Clusterable*> &points,
    std::vector<Clusterable*> *clusters_out,
    std::vector<int32> *assignments_out);


}  // end namespace kaldi

#endif  // KALDI_SEGMENTER_ITERATIVE_BOTTOM_UP_CLUSTER_H_
