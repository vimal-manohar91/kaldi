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
  void Check() const;
  
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

// This class is used for splitting something of type
// DiscriminativeSupervision into
// multiple pieces corresponding to different frame-ranges.
class DiscriminativeSupervisionSplitter {
 public:
  typedef fst::ArcTpl<LatticeWeight> LatticeArc;
  typedef fst::VectorFst<LatticeArc> Lattice;
 
  DiscriminativeSupervisionSplitter(const DiscriminativeSupervision &supervision);
  
  CreateSplit(int32 begin_frame, int32 frames_per_seq, 
              DiscriminativeSupervision *supervision);

 private:
  // Extracts a frame range of the supervision into 'supervision'.  
  void GetFrameRange(int32 begin_frame, int32 frames_per_sequence,
                     DiscriminativeSupervision *supervision) const;

  // Creates an output lattice covering frames begin_frame <= t < end_frame,
  // assuming that the corresponding state-range that we need to
  // include, begin_state <= s < end_state has been included.
  // (note: the output lattice will also have two special initial and final
  // states).  Does not do the post-processing (RmEpsilon, Determinize,
  // TopSort on the result).  See code for details.
  void CreateRangeLattice(const Lattice &in_lat,
                          const std::vector<int32> state_times,
                          int32 begin_frame, int32 end_frame,
                          Lattice *out_lat) const;

  const DiscriminativeSupervision &supervision_;

  struct LatticeInfo {
    std::vector<BaseFloat> alpha_p;
    std::vector<BaseFloat> beta_p;
    std::vector<BaseFloat> alpha_r;
    std::vector<BaseFloat> beta_r;
    std::vector<int32> state_times;

    bool Check() {
      state_times.size() == alpha_p.size();
      state_times.size() == beta_p.size();
      state_times.size() == alpha_r.size();
      state_times.size() == beta_r.size();
    } const;
  };

  LatticeInfo num_lat_scores_;
  LatticeInfo den_lat_scores_;

  Lattice num_lat_;
  Lattice den_lat_;
  bool num_lat_present_;

  ComputeLatticeScores(const Lattice &lat, LatticeInfo *scores) const;
};

/// This function appends a list of supervision objects to create what will
/// usually be a single such object, but if the weights and num-frames are not
/// all the same it will only append Supervision objects where successive ones
/// have the same weight and num-frames, and if 'compactify' is true.  The
/// normal use-case for this is when you are combining neural-net examples for
/// training; appending them like this helps to simplify the decoding process.

/// This function will crash if the values of label_dim in the inputs are not
/// all the same.
void AppendSupervision(const std::vector<const Supervision*> &input,
                       bool compactify,
                       std::vector<Supervision> *output_supervision);

/// This function helps you to pseudo-randomly split a sequence of length 'num_frames',
/// interpreted as frames 0 ... num_frames - 1, into pieces of length exactly
/// 'frames_per_range', to be used as examples for training.  Because frames_per_range
/// may not exactly divide 'num_frames', this function will leave either small gaps or
/// small overlaps in pseudo-random places.
/// The output 'range_starts' will be set to a list of the starts of ranges, the
/// output ranges are of the form
/// [ (*range_starts)[i] ... (*range_starts)[i] + frames_per_range - 1 ].
void SplitIntoRanges(int32 num_frames,
                     int32 frames_per_range,
                     std::vector<int32> *range_starts);

// This utility function is not used directly in the 'discriminative' code.  It
// is used to get weights for the derivatives, so that we don't doubly train on
// some frames after splitting them up into overlapping ranges of frames.  The
// input 'range_starts' will be obtained from 'SplitIntoRanges', but the
// 'range_length', which is a length in frames, may be longer than the one
// supplied to SplitIntoRanges, due the 'overlap'.  (see the calling code...  if
// we want overlapping ranges, we get it by 'faking' the input to
// SplitIntoRanges).
//
// The output vector 'weights' will be given the same dimension as
// 'range_starts'.  By default the output weights in '*weights' will be vectors
// of all ones, of length equal to 'range_length', and '(*weights)[i]' represents
// the weights given to frames numbered
//   t = range_starts[i] ... range_starts[i] + range_length - 1.
// If these ranges for two successive 'i' values overlap, then we
// reduce the weights to ensure that no 't' value gets a total weight
// greater than 1.  We do this by dividing the overlapped region
// into three approximately equal parts, and giving the left part
// to the left range; the right part to the right range; and
// in between, interpolating linearly.
void GetWeightsForRanges(int32 range_length,
                         const std::vector<int32> &range_starts,
                         std::vector<Vector<BaseFloat> > *weights);

typedef TableWriter<KaldiObjectHolder<DiscriminativeSupervision> > DiscriminativeSupervisionWriter;
typedef SequentialTableReader<KaldiObjectHolder<DiscriminativeSupervision> > SequentialDiscriminativeSupervisionReader;
typedef RandomAccessTableReader<KaldiObjectHolder<DiscriminativeSupervision> > RandomAccessDiscriminativeSupervisionReader;

} 
}

#endif // KALDI_NNET3_DISCRIMINATIVE_SUPERVISION_H
