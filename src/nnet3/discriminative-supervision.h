// nnet3/discriminative-supervision.h

// Copyright 2012-2015  Johns Hopkins University (author: Daniel Povey)
//           2014-2015  Vimal Manohar

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

#ifndef KALDI_NNET3_DISCRIMINATIVE_SUPERVISION_H
#define KALDI_NNET3_DISCRIMINATIVE_SUPERVISION_H

#include "util/table-types.h"
#include "hmm/posterior.h"
#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {
namespace nnet3 {

// struct DiscriminativeSupervision is the fully-processed information for
// a whole utterance or (after splitting) part of an utterance. 
struct DiscriminativeSupervision {
  // The weight we assign to this example;
  // this will typically be one, but we include it
  // for the sake of generality.  
  BaseFloat weight; 
  
  // num_sequences will be 1 if you create a DiscriminativeSupervision object from a single
  // lattice or alignment, but if you combine multiple DiscriminativeSupervision objects
  // the 'num_sequences' is the number of objects that were combined (the
  // lattices get appended).
  int32 num_sequences;

  // the number of frames in each sequence of appended objects.  num_frames *
  // num_sequences must equal the path length of any path in the lattices.
  // Technically this information is redundant with the lattices, but it's convenient
  // to have it separately.
  int32 frames_per_sequence;
  
  // the maximum possible value of the labels in 'fst' (which go from 1 to
  // label_dim).  This should equal the NumPdfs() in the TransitionModel object.
  // Included to avoid training on mismatched egs.
  int32 label_dim;

  // The numerator alignment
  std::vector<int32> num_ali;
  
  // Alternate alignment for debugging purposes; in the case of 
  // semi-supervised training, this could hold the oracle alignment.
  std::vector<int32> oracle_ali;
  
  // Frame weights, usually a value between 0 and 1 to indicate the 
  // contribution of each frame to the objective function value.
  // The default weight (when this vector is empty) is 1 for each frame.
  std::vector<BaseFloat> weights;
  
  // Note: any acoustic
  // likelihoods in the lattices will be
  // recomputed at the time we train.

  // Indicates whether a numerator lattice is present.
  bool num_lat_present;
  
  // The numerator lattice
  CompactLattice num_lat;
  
  // The denominator lattice.  
  CompactLattice den_lat; 
  
  DiscriminativeSupervision(): weight(1.0), num_sequences(1),
                               frames_per_sequence(-1), label_dim(-1),
                               num_lat_present(false) { }

  DiscriminativeSupervision(const DiscriminativeSupervision &other);

  void Swap(DiscriminatveSupervision *other);

  bool operator == (const DiscriminativeSupervision &other) const;
  
  // This function checks that this supervision object satifsies some
  // of the properties we expect of it, and calls KALDI_ERR if not.
  void Check(const TransitionModel &trans_model) const;
  
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};

/// This creates supervision alignment, denominator lattice and 
/// optionally oracle alignment, frame weights 
/// as required from discriminative objective functions.
bool LatticeToDiscriminativeSupervision(
    const std::vector<int32> &alignment,
    const CompactLattice &clat,
    BaseFloat weight,
    DiscriminativeSupervision *supervision,
    const Vector<BaseFloat> *weights = NULL,
    const std::vector<int32> *oracle_alignment = NULL);

/// This constructor is similar to the above but also uses a numerator
/// lattice to create discriminative example.
bool LatticeToDiscriminativeSupervision(
    const std::vector<int32> &alignment,
    const CompactLattice &num_clat,
    const CompactLattice &clat,
    BaseFloat weight,
    DiscriminativeSupervision *supervision,
    const Vector<BaseFloat> *weights = NULL,
    const std::vector<int32> *oracle_alignment = NULL);

/// This constructor is similar to the above but also uses a numerator
/// posterior to create discriminative example.
bool LatticeToDiscriminativeSupervision(
    const std::vector<int32> &alignment,
    const Posterior &num_post,
    int32 dim,
    const CompactLattice &clat,
    BaseFloat weight,
    DiscriminativeSupervision *supervision,
    const Vector<BaseFloat> *weights = NULL,
    const std::vector<int32> *oracle_alignment = NULL);

} 
}

#endif // KALDI_NNET3_DISCRIMINATIVE_SUPERVISION_H
