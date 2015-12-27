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
namespace discriminative {

/*
  This file contains some declarations relating to the object we use to
  encode the supervision information for sequence training
*/

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
  
  // the maximum possible value of the labels in 'lattices' (which go from 1 to
  // label_dim).  
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
  Lattice num_lat;
  
  // The denominator lattice.  
  Lattice den_lat; 
  
  DiscriminativeSupervision(): weight(1.0), num_sequences(1),
                               frames_per_sequence(-1), label_dim(-1),
                               num_lat_present(false) { }

  DiscriminativeSupervision(const DiscriminativeSupervision &other);

  void Swap(DiscriminativeSupervision *other);

  bool operator == (const DiscriminativeSupervision &other) const;
  
  // This function checks that this supervision object satifsies some
  // of the properties we expect of it, and calls KALDI_ERR if not.
  void Check() const;
  
  int32 NumFrames() const { return num_sequences * frames_per_sequence; }

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};

/// This creates supervision alignment, denominator lattice and 
/// optionally oracle alignment, frame weights 
/// as required from discriminative objective functions.
bool LatticeToDiscriminativeSupervision(
    const std::vector<int32> &alignment,
    const Lattice &lat,
    BaseFloat weight,
    DiscriminativeSupervision *supervision,
    const Vector<BaseFloat> *weights = NULL,
    const std::vector<int32> *oracle_alignment = NULL);

/// This constructor is similar to the above but also uses a numerator
/// lattice to create discriminative example.
bool LatticeToDiscriminativeSupervision(
    const std::vector<int32> &alignment,
    const Lattice &num_lat,
    const Lattice &den_lat,
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
    const Lattice &lat,
    BaseFloat weight,
    DiscriminativeSupervision *supervision,
    const Vector<BaseFloat> *weights = NULL,
    const std::vector<int32> *oracle_alignment = NULL);

// This class is used for splitting something of type
// DiscriminativeSupervision into
// multiple pieces corresponding to different frame-ranges.
class DiscriminativeSupervisionSplitter {
 public:
  typedef fst::ArcTpl<LatticeWeight> LatticeArc;
  typedef fst::VectorFst<LatticeArc> Lattice;
 
  DiscriminativeSupervisionSplitter(const DiscriminativeSupervision &supervision);

  struct LatticeInfo {
    std::vector<double> alpha_p;
    std::vector<double> beta_p;
    std::vector<double> alpha_r;
    std::vector<double> beta_r;
    std::vector<int32> state_times;

    void Check() const {
      KALDI_ASSERT(state_times.size() == alpha_p.size() &&
          state_times.size() == beta_p.size() &&
          state_times.size() == alpha_r.size() &&
          state_times.size() == beta_r.size());
    } 
  };
  
  // Extracts a frame range of the supervision into 'supervision'.  
  void GetFrameRange(int32 begin_frame, int32 frames_per_sequence,
                     DiscriminativeSupervision *supervision) const;

 private:

  // Creates an output lattice covering frames begin_frame <= t < end_frame,
  // assuming that the corresponding state-range that we need to
  // include, begin_state <= s < end_state has been included.
  // (note: the output lattice will also have two special initial and final
  // states).  Does not do the post-processing (RmEpsilon, Determinize,
  // TopSort on the result).  See code for details.
  void CreateRangeLattice(const Lattice &in_lat,
                          const LatticeInfo &scores,
                          int32 begin_frame, int32 end_frame,
                          Lattice *out_lat) const;

  const DiscriminativeSupervision &supervision_;

  LatticeInfo num_lat_scores_;
  LatticeInfo den_lat_scores_;

  Lattice num_lat_;
  Lattice den_lat_;
  bool num_lat_present_;

  void ComputeLatticeScores(const Lattice &lat, LatticeInfo *scores) const;
};

/// This function appends a list of supervision objects to create what will
/// usually be a single such object, but if the weights and num-frames are not
/// all the same it will only append Supervision objects where successive ones
/// have the same weight and num-frames, and if 'compactify' is true.  The
/// normal use-case for this is when you are combining neural-net examples for
/// training; appending them like this helps to simplify the decoding process.

/// This function will crash if the values of label_dim in the inputs are not
/// all the same.
void AppendSupervision(const std::vector<const DiscriminativeSupervision*> &input,
                       bool compactify,
                       std::vector<DiscriminativeSupervision> *output_supervision);

// Extend a lattice *lat by appending a lattice src_lat at the end of it
void AppendLattice(Lattice *lat, const Lattice &src_lat);

typedef TableWriter<KaldiObjectHolder<DiscriminativeSupervision> > DiscriminativeSupervisionWriter;
typedef SequentialTableReader<KaldiObjectHolder<DiscriminativeSupervision> > SequentialDiscriminativeSupervisionReader;
typedef RandomAccessTableReader<KaldiObjectHolder<DiscriminativeSupervision> > RandomAccessDiscriminativeSupervisionReader;

} // namespace discriminative
} // namespace kaldi

#endif // KALDI_NNET3_DISCRIMINATIVE_SUPERVISION_H
