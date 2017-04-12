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
#include "segmenter/gaussian-stats-clusterable.h"

#ifndef KALDI_SEGMENTER_SEGMENTATION_CLUSTER_UTILS_H_
#define KALDI_SEGMENTER_SEGMENTATION_CLUSTER_UTILS_H_

namespace kaldi { 

struct SegmentClusteringOptions {
  int32 length_tolerance;
  int32 window_size;
  bool merge_only_overlapping_segments;
  BaseFloat statistics_scale;
  GaussianStatsOptions gaussian_stats_opts;

  SegmentClusteringOptions() : 
    length_tolerance(2), window_size(150),  
    merge_only_overlapping_segments(false),
    statistics_scale(1.0) { }

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

