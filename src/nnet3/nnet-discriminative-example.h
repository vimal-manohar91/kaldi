// nnet3/nnet-discriminative-example.h

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

#ifndef KALDI_NNET3_NNET_DISCRIMINATIVE_EXAMPLE_H_
#define KALDI_NNET3_NNET_DISCRIMINATIVE_EXAMPLE_H_

#include "nnet3/nnet-nnet.h"
#include "util/table-types.h"
#include "nnet3/discriminative-supervision.h"
#include "hmm/posterior.h"
#include "hmm/transition-model.h"

namespace kaldi {
namespace nnet3 {

// Glossary: mmi = Maximum Mutual Information,
//          mpfe = Minimum Phone Frame Error
//          smbr = State-level Minimum Bayes Risk
//           nce = Negative Conditional Entropy
//         esmbr = Extended State-level Minimum Bayes Risk
//         empfe = Extended Phone Frame Error

// esmbr and empfe use either posteriors or lattices for 
// numerator lattices, whereas smbr and mpfe use alignments

// This file relates to the creation of examples for discriminative training

struct NnetDiscriminativeSupervision {
  /// the name of the output in the neural net; in simple setups it
  /// will just be "output".
  std::string name;
  
  /// The indexes that the output corresponds to.  The size of this vector will
  /// be equal to supervision.num_sequences * supervision.frames_per_sequence.
  /// Be careful about the order of these indexes-- it is a little confusing.
  /// The indexes in the 'index' vector are ordered as: (frame 0 of each sequence);
  /// (frame 1 of each sequence); and so on.  But in the 'supervision' object,
  /// the FST contains (sequence 0; sequence 1; ...).  So reordering is needed.
  /// This is done for efficiency in the denominator computation (it helps memory
  /// locality), as well as to match the ordering inside the neural net.
  std::vector<Index> indexes;

  /// The supervision object, containing the numerator and denominator 
  /// lattices.
  DiscriminativeSupervision supervision;

  /// This is a vector of per-frame weights, required to be between 0 and 1,
  /// that is applied to the derivative during training (but not during model
  /// combination, where the derivatives need to agree with the computed objf
  /// values for the optimization code to work).  The reason for this is to more
  /// exactly handle edge effects and to ensure that no frames are
  /// 'double-counted'.  The order of this vector corresponds to the order of
  /// the 'indexes' (i.e. all the first frames, then all the second frames,
  /// etc.)
  /// If this vector is empty it means we're not applying per-frame weights,
  /// so it's equivalent to a vector of all ones.  This vector is written
  /// to disk compactly as unsigned char.
  Vector<BaseFloat> deriv_weights;
  
  // Use default assignment operator
  NnetDiscriminativeSupervision() { }

  /// Initialize the object from an object of type chain::Supervision, and some
  /// extra information.  Note: you probably want to set 'name' to "output".
  /// 'first_frame' will often be zero but you can choose (just make it
  /// consistent with how you numbered your inputs), and 'frame_skip' would be 1
  /// in a vanilla setup, but we plan to try setups where the output periodicity
  /// is slower than the input, so in this case it might be 2 or 3.
  NnetDiscriminativeSupervision(const std::string &name,
                                const DiscriminativeSupervision &supervision,
                                const Vector<BaseFloat> &deriv_weights,
                                int32 first_frame,
                                int32 frame_skip);

  NnetDiscriminativeSupervision(const NnetDiscriminativeSupervision &other);

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);
  
  void Swap(NnetDiscriminativeSupervision *other);

  void CheckDim() const;
  
  bool operator == (const NnetDiscriminativeSupervision &other) const;
};

/// NnetDiscriminativeExample is like NnetExample, but specialized for 
/// sequence training.
struct NnetDiscriminativeExample {

  /// 'inputs' contains the input to the network-- normally just it has just one
  /// element called "input", but there may be others (e.g. one called
  /// "ivector")...  this depends on the setup.
  std::vector<NnetIo> inputs;

  /// 'outputs' contains the sequence output supervision.  There will normally
  /// be just one member with name == "output".
  std::vector<NnetDiscriminativeSupervision> outputs;

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  void Swap(NnetDiscriminativeExample *other);

  // Compresses the input features (if not compressed)
  void Compress();

  NnetDiscriminativeExample() { }

  NnetDiscriminativeExample(const NnetDiscriminativeExample &other);

  bool operator == (const NnetChainExample &other) const {
    return inputs == other.inputs && outputs == other.outputs;
  }
};

/** 
  Appends the given vector of examples (which must be non-empty) into 
  a single output example.
  Intended to be used when forming minibatches for neural net training. If 
  'compress' it compresses the output features (recommended to save disk
  space).

  Note: the input is left as it was at the start, but it is temporarily
  changed inside the function; this is a trick to allow us to use the
  MergeExamples() routine while avoiding having to rewrite code.
*/
void MergeDiscriminativeExamples(
    bool compress,
    const std::vector<const NnetDiscriminativeExample*> &input,
    NnetDiscriminativeExample *output);

/** Shifts the time-index t of everything in the input of "eg" by adding
    "t_offset" to all "t" values-- but excluding those with names listed in
    "exclude_names", e.g.  "ivector".  This might be useful if you are doing
    subsampling of frames at the output, because shifted examples won't be quite
    equivalent to their non-shifted counterparts.  "exclude_names" is a vector
    of names of nnet inputs that we avoid shifting the "t" values of-- normally
    it will contain just the single string "ivector" because we always leave t=0
    for any ivector.

    Note: input features will be shifted by 'frame_shift', and indexes in the
    supervision in (eg->output) will be shifted by 'frame_shift' rounded to the
    closest multiple of the frame subsampling factor (e.g. 3).  The frame
    subsampling factor is worked out from the time spacing between the indexes
    in the output.  */
void ShiftDiscriminativeExampleTimes(int32 frame_shift,
                                    const std::vector<std::string> &exclude_names,
                                    NnetDiscriminativeExample *eg);

/**
   This sets to zero any elements of 'egs->outputs[*].deriv_weights' that correspond
   to frames within the first or last 'truncate' frames of the sequence (e.g. you could
   set 'truncate=5' to set zero deriv-weight for the first and last 5 frames of the
   sequence).
 */
void TruncateDerivWeights(int32 truncate,
                          NnetDiscriminativeExample *eg);

/**  This function takes a NnetDiscriminativeExample and produces a 
     ComputationRequest.
     Assumes you don't want the derivatives w.r.t. the inputs; if you do, you
     can create the ComputationRequest manually.  Assumes that if
     need_model_derivative is true, you will be supplying derivatives w.r.t. all
     outputs.
*/
void GetDiscriminativeComputationRequest(const Nnet &nnet,
                                         const NnetDiscriminativeExample &eg,
                                         bool need_model_derivative,
                                         bool store_component_stats,
                                         ComputationRequest *computation_request);

/**
   Given a discriminative training example, this function works out posteriors
   at the pdf level (note: these are "discriminative-training posteriors" that
   may be positive or negative.  The denominator lattice "den_lat" in the
   example "eg" should already have had acoustic-rescoring done so that its
   acoustic probs are up to date, and any acoustic scaling should already have
   been applied.

   "criterion" may be one of "mmi", "mpfe", "smbr", "nce", "empfe" and "esmbr".  
   If criterion is "mmi", "drop_frames" means we don't include derivatives for
   frames where the numerator pdf is not in the denominator lattice.

   if "one_silence_class" is true you can get a newer behavior for MPE/SMBR
   which will tend to reduce insertions.

   "silence_phones" is a list of silence phones (this is only relevant for mpfe
   or smbr, if we want to treat silence specially).
 */
void ExampleToPdfPost(
    const TransitionModel &tmodel,
    const std::vector<int32> &silence_phones,
    std::string criterion,
    bool drop_frames,
    bool one_silence_class,
    const NnetDiscriminativeExample &eg,
    Posterior *post);

/**
   This function is used in code that tests the functionality that we provide
   here, about splitting and excising nnet examples.  It adds to a "hash
   function" that is a function of a set of examples; the hash function is of
   dimension (number of pdf-ids x features dimension).  The hash function
   consists of the (denominator - numerator) posteriors over pdf-ids, times the
   average over the context-window (left-context on the left, right-context on
   the right), of the features.  This is useful because the various
   manipulations we do are supposed to preserve this, and if there is a bug
   it will most likely cause the hash function to change.

   This function will resize the matrix if it is empty.

   Any acoustic scaling of the lattice should be done before you call this
   function.
   
   "criterion" may be one of "mmi", "mpfe", "smbr", "nce", "empfe" and "esmbr".  

   You should set drop_frames to true if you are doing MMI with drop-frames
   == true.  Then it will not compute the hash for frames where the numerator
   pdf-id is not in the denominator lattice.

   You can set one_silence_class to true for a newer optional behavior that will
   reduce insertions in the trained model (or false for the traditional
   behavior).

   The function will also accumulate the total numerator and denominator weights
   used as num_weight and den_weight, for an additional diagnostic, and the total
   number of frames, as tot_t.
*/
void UpdateHash(
    const TransitionModel &tmodel,
    const NnetDiscriminativeExample &eg,
    std::string criterion,
    bool drop_frames,
    bool one_silence_class,
    Matrix<double> *hash,
    double *num_weight,
    double *den_weight,
    double *tot_t);


typedef TableWriter<KaldiObjectHolder<NnetDiscriminativeExample > > NnetDiscriminativeExampleWriter;
typedef SequentialTableReader<KaldiObjectHolder<NnetDiscriminativeExample > > SequentialNnetDiscriminativeExampleReader;
typedef RandomAccessTableReader<KaldiObjectHolder<NnetDiscriminativeExample > > RandomAccessNnetDiscriminativeExampleReader;

} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_DISCRIMINATIVE_EXAMPLE_H_

