// segmenter/iterative-bottom-up-cluster-inl.h

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

#ifndef KALDI_SEGMENTER_ITERATIVE_BOTTOM_UP_CLUSTER_INL_H_
#define KALDI_SEGMENTER_ITERATIVE_BOTTOM_UP_CLUSTER_INL_H_

#include "itf/clusterable-itf.h"
#include "tree/cluster-utils.h"

namespace kaldi {
  
void FlattenCompartments(
    const std::vector<std::vector<Clusterable *> > 
    &compartmentalized_clusters,
    std::vector<Clusterable *> *clusterables, 
    std::vector<std::vector<int32> > *compartmentalized_assignment2id);

}  // end namespace kaldi

#endif  // KALDI_SEGMENTER_ITERATIVE_BOTTOM_UP_CLUSTER_INL_H_
