// nnet2/nnet-compute-sequence.h

// Copyright 2012-2013  Johns Hopkins University (author: Daniel Povey)
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

#ifndef KALDI_NNET2_NNET_COMPUTE_DISCRIMINATIVE_H_
#define KALDI_NNET2_NNET_COMPUTE_DISCRIMINATIVE_H_

#include "nnet2/am-nnet.h"
#include "nnet2/nnet-example.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include "base/kaldi-types-extra.h"

namespace kaldi {
namespace nnet2 {
 
typedef SignedLogReal<double> SignedLogDouble;

/* This header provides functionality for doing model updates, and computing
   gradients, using sequence training objective functions 
   for both supervised (MPFE, SMBR, MMI) and unsupervised (EMPFE, ESMBR, NCE) 
   settings.
   We use the DiscriminativeNnetExample defined in nnet-example.h.
*/

struct NnetDiscriminativeUpdateOptions {
  std::string criterion; // "mmi" or "mpfe" or "smbr" or "nce" or "empfe" or "esmbr"
  BaseFloat acoustic_scale; // e.g. 0.1
  bool drop_frames; // for MMI, true if we ignore frames where alignment
                    // pdf-id is not in the lattice.
  bool one_silence_class;  // Affects MPFE/SMBR/EMPFE/ESMBR>
  BaseFloat deletion_penalty;     // e.g. 0.1. Affects ESMBR and EMPFE.
  BaseFloat boost; // for MMI, boosting factor (would be Boosted MMI)... e.g. 0.1.

  std::string silence_phones_str; // colon-separated list of integer ids of silence phones,
                                  // for MPFE/SMBR/EMPFE/ESMBR only.
  BaseFloat weight_threshold; // e.g. 0.0
  
  NnetDiscriminativeUpdateOptions(): criterion("smbr"), acoustic_scale(0.1),
                                     drop_frames(false),
                                     one_silence_class(false),
                                     deletion_penalty(0.0),
                                     boost(0.0), weight_threshold(0.0) { }

  void Register(OptionsItf *opts) {
    opts->Register("criterion", &criterion, "Criterion, 'mmi'|'mpfe'|'smbr'|'nce'|'empfe'|'esmbr', "
                   "determines the objective function to use.  Should match "
                   "option used when we created the examples.");
    opts->Register("acoustic-scale", &acoustic_scale, "Weighting factor to "
                   "apply to acoustic likelihoods.");
    opts->Register("drop-frames", &drop_frames, "For MMI, if true we drop frames "
                   "with no overlap of num and den frames");
    opts->Register("boost", &boost, "Boosting factor for boosted MMI (e.g. 0.1)");
    opts->Register("one-silence-class", &one_silence_class, "If true, newer "
                   "behavior which will tend to reduce insertions.");
    opts->Register("deletion-penalty", &deletion_penalty, "Penalize deletions "
                 "by favoring paths that don't have deletions.");
    opts->Register("silence-phones", &silence_phones_str,
                   "For MPFE or SMBR, colon-separated list of integer ids of "
                   "silence phones, e.g. 1:2:3");
    opts->Register("weight-threshold", &weight_threshold, 
                 "Ignore frames below a confidence threshold");
    
  }
};

struct NnetDiscriminativeStats {
  double tot_t; // total number of frames
  double tot_t_weighted; // total number of frames times weight.
  double tot_objf;      // for MMI, the (weighted) denominator likelihood; for
                        // everything else, the objective function.
  double tot_gradients; // for NCE, the gradients, for everything else 0
  double tot_num_count; // total count of numerator posterior for everything but NCE
  double tot_den_count; // total count of denominator posterior for everything but NCE
  double tot_num_objf;  // for MMI, the (weighted) numerator likelihood; for
                        // everything else 0
  bool store_gradients;
  bool store_logit_stats;

  CuVector<double> gradients;
  CuVector<double> output;
  CuVector<double> logit_gradients;
  CuVector<double> logit;
  CuVector<double> indication_counts;

  NnetDiscriminativeStats(int32 num_pdfs) { 
    std::memset(this, 0, sizeof(*this)); 
    gradients.Resize(num_pdfs); 
    output.Resize(num_pdfs);
    indication_counts.Resize(num_pdfs);
    store_gradients = true;
    store_logit_stats = true;
  }

  NnetDiscriminativeStats() {
    std::memset(this, 0, sizeof(*this));
    store_gradients = false;
    store_logit_stats = false;
  }

  void Print(string criterion, bool print_gradients = false, bool print_post = false) const;
  void PrintPost(int32 pdf_id) const;
  void Add(const NnetDiscriminativeStats &other);
};

/*
  This class does the forward and possibly backward computation for (typically)
  a whole utterance of contiguous features.  You'll instantiate one of
  these classes each time you want to do this computation.
*/
class NnetDiscriminativeUpdater {
 public:
  NnetDiscriminativeUpdater(const AmNnet &am_nnet,
                      const TransitionModel &tmodel,
                      const NnetDiscriminativeUpdateOptions &opts,
                      const DiscriminativeNnetExample &eg,
                      Nnet *nnet_to_update,
                      NnetDiscriminativeStats *stats);

  SignedLogDouble Update() {
    Propagate();
    SignedLogDouble objf = LatticeComputations();
    if (nnet_to_update_ != NULL)
      Backprop();
    return objf;
  }
  
  /// The forward-through-the-layers part of the computation.
  void Propagate();  

  /// Does the parts between Propagate() and Backprop(), that
  /// involve forward-backward over the lattice.
  SignedLogDouble LatticeComputations();

  void Backprop();

  /// Assuming the lattice already has the correct scores in
  /// it, this function does the MPE or MMI forward-backward
  /// and puts the resulting discriminative posteriors (which
  /// may have positive or negative weight) into "post".
  /// It returns, for MPFE/SMBR, the objective function, or
  /// for MMI, the negative of the denominator-lattice log-likelihood.
  SignedLogDouble GetDerivativesWrtActivations(Posterior *post);
  
  SubMatrix<BaseFloat> GetInputFeatures() const;

  CuMatrixBase<BaseFloat> &GetOutput() { return forward_data_.back(); }

  static inline Int32Pair MakePair(int32 first, int32 second) {
    Int32Pair ans;
    ans.first = first;
    ans.second = second;
    return ans;
  }

  const Lattice& GetLattice() const { return lat_; }
  void SetLattice(Lattice &lat) { lat_ = lat; }

  const Lattice& GetNumeratorLattice() const { return num_lat_; }
  void SetNumeratorLattice(Lattice &num_lat) { num_lat_ = num_lat; }
 private:
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;

  
  const AmNnet &am_nnet_;
  const TransitionModel &tmodel_;
  const NnetDiscriminativeUpdateOptions &opts_;
  const DiscriminativeNnetExample &eg_;
  Nnet *nnet_to_update_; // will equal am_nnet_.GetNnet(), in SGD case, or
                         // another Nnet, in gradient-computation case, or
                         // NULL if we just need the objective function.
  NnetDiscriminativeStats *stats_; // the objective function, etc.
  std::vector<ChunkInfo> chunk_info_out_; 
  // forward_data_[i] is the input of the i'th component and (if i > 0)
  // the output of the i-1'th component.
  std::vector<CuMatrix<BaseFloat> > forward_data_; 
  Lattice lat_; // we convert the CompactLattice in the eg, into Lattice form.
  Lattice num_lat_; // we convert the numerator CompactLattice in the eg, into Lattice form.
  CuMatrix<BaseFloat> backward_data_;
  std::vector<int32> silence_phones_; // derived from opts_.silence_phones_str
    
};

/** Does the neural net computation, lattice forward-backward, and backprop,
    for either the MMI, MPFE or SMBR objective functions.
    If nnet_to_update == &(am_nnet.GetNnet()), then this does stochastic
    gradient descent, otherwise (assuming you have called SetZero(true)
    on *nnet_to_update) it will compute the gradient on this data.
    If nnet_to_update_ == NULL, no backpropagation is done.
    
    Note: we ignore any existing acoustic score in the lattice of "eg".

    For display purposes you should normalize the sum of this return value by
    dividing by the sum over the examples, of the number of frames
    (num_ali.size()) times the weight.

    Something you need to be careful with is that the occupation counts and the
    derivative are, following tradition, missing a factor equal to the acoustic
    scale.  So you need to multiply them by that scale if you plan to do
    something like L-BFGS in which you look at both the derivatives and function
    values.  */

SignedLogDouble NnetDiscriminativeUpdate(const AmNnet &am_nnet,
                              const TransitionModel &tmodel,
                              const NnetDiscriminativeUpdateOptions &opts,
                              const DiscriminativeNnetExample &eg,
                              Nnet *nnet_to_update = NULL,
                              NnetDiscriminativeStats *stats = NULL);

} // namespace nnet2
} // namespace kaldi

#endif // KALDI_NNET2_NNET_COMPUTE_DISCRIMINATIVE_H_
