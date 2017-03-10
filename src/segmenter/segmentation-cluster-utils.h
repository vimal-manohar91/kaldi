// segmenter/segmentation-cluster-utils.h

// Copyright 2017   Vimal Manohar (Johns Hopkins University)

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
#include "segmenter/segmentation.h"

#ifndef KALDI_SEGMENTER_SEGMENTATION_CLUSTER_UTILS_H_
#define KALDI_SEGMENTER_SEGMENTATION_CLUSTER_UTILS_H_

namespace kaldi { 

struct SegmentClusteringOptions {
  int32 length_tolerance;
  int32 window_size;
  bool use_full_covar;
  std::string distance_metric;
  BaseFloat bic_penalty;
  BaseFloat var_floor;
  BaseFloat threshold;
  bool merge_only_overlapping_segments;

  SegmentClusteringOptions() : 
    length_tolerance(2), window_size(150),  
    use_full_covar(false), distance_metric("kl2"),
    bic_penalty(2.0), var_floor(0.01), threshold(0.0),
    merge_only_overlapping_segments(false) { }

  void Register(OptionsItf *opts);
};

int32 SplitByChangePoints(const SegmentClusteringOptions &opts,
                          const MatrixBase<BaseFloat> &feats,
                          const segmenter::Segmentation &segmentation,
                          segmenter::Segmentation *out_segmentation);

int32 ClusterAdjacentSegments(const SegmentClusteringOptions &opts,
                              const MatrixBase<BaseFloat> &feats,
                              segmenter::Segmentation *segmentation);

}

#endif  // KALDI_SEGMENTER_SEGMENTATION_CLUSTER_UTILS_H_

