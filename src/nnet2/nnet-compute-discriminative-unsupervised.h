// nnet2/nnet-compute-discriminative-unsupervised.h

// Copyright 2014   Vimal Manohar

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

#ifndef KALDI_NNET2_NNET_COMPUTE_DISCRIMINATIVE_UNSUPERVISED_H_
#define KALDI_NNET2_NNET_COMPUTE_DISCRIMINATIVE_UNSUPERVISED_H_

#include "nnet2/am-nnet.h"
#include "nnet2/nnet-example.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include "base/kaldi-types-extra.h"

namespace kaldi {
namespace nnet2 {

typedef SignedLogReal<double> SignedLogDouble;

/* This head provides functionality for doing model updates, and computing
   gradients using discriminative semi-supervised objective functions.
   We use the DiscriminativeUnsupervisedNnetExample defined in nnet-example.h.
*/

struct NnetDiscriminativeUnsupervisedUpdateOptions {
  BaseFloat acoustic_scale; // e.g. 0.1

  NnetDiscriminativeUnsupervisedUpdateOptions(): acoustic_scale(0.1) { }

  void Register(OptionsItf *po) {
    po->Register("acoustic-scale", &acoustic_scale, "Weighting factor to "
                 "apply to acoustic likelihoods.");
  }
};

struct NnetDiscriminativeUnsupervisedStats {
  double tot_t;          // total number of frames
  double tot_t_weighted; // total number of frames times weight.
  double tot_objf;       // Negative Conditional Entropy (NCE) objective function
  double tot_gradients;
  bool store_gradients;
  Vector<double> gradients;

  NnetDiscriminativeUnsupervisedStats(int32 num_pdfs) { 
    std::memset(this, 0, sizeof(*this)); 
    gradients.Resize(num_pdfs); 
    store_gradients = true;
  }

  NnetDiscriminativeUnsupervisedStats() {
    std::memset(this, 0, sizeof(*this));
    store_gradients = false;
  }

  void Print() const;
  void PrintPost(int32 pdf_id) const;
  void Add(const NnetDiscriminativeUnsupervisedStats &other);
};

/*
  This class does the forward and possibly backward computation for (typically)
  a whole utterance of contiguous features and accumulates per node
  both p_a and r_a=p_a log(p_a), where p_a is the acoustic likelihood score on arc a.  
  You'll instantiate one of these classes each time you want to do this computation.
*/
class NnetDiscriminativeUnsupervisedUpdater {
 public:
  NnetDiscriminativeUnsupervisedUpdater(const AmNnet &am_nnet,
                                        const TransitionModel &tmodel,
                                        const NnetDiscriminativeUnsupervisedUpdateOptions &opts,
                                        const DiscriminativeUnsupervisedNnetExample &eg,
                                        Nnet *nnet_to_update,
                                        NnetDiscriminativeUnsupervisedStats *stats);

  SignedLogDouble Update() {
    Propagate();
    SignedLogDouble objf = LatticeComputations();
    if (nnet_to_update_ != NULL)
      Backprop();
    return objf;
  }
  
  /// The forward-through-the-layers part of the computation
  void Propagate();

  /// Does the parts between Propagate() and Backprop(), that
  /// involve forward-backward over the lattice
  SignedLogDouble LatticeComputations();

  void Backprop();

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

 private:
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;

  
  const AmNnet &am_nnet_;
  const TransitionModel &tmodel_;
  const NnetDiscriminativeUnsupervisedUpdateOptions &opts_;
  const DiscriminativeUnsupervisedNnetExample &eg_;
  Nnet *nnet_to_update_; // will equal am_nnet_.GetNnet(), in SGD case, or
                         // another Nnet, in gradient-computation case, or
                         // NULL if we just need the objective function.
  NnetDiscriminativeUnsupervisedStats *stats_; // the objective function, etc.
  std::vector<ChunkInfo> chunk_info_out_; 

  // forward_data_[i] is the input of the i'th component and (if i > 0)
  // the output of the i-1'th component.
  
  std::vector<CuMatrix<BaseFloat> > forward_data_;
  Lattice lat_; // we convert the CompactLattice in the eg, into Lattice form.
  CuMatrix<BaseFloat> backward_data_;
};


/** Does the neural net computation, lattice forward-backward with 
    conditional entropy computation along with probabilities, and backprop
    for NCE objective function.
    If nnet_to_update == &(am_nnet.GetNnet(), then this does stochastic 
    gradient descent, otherwise (assuming you have called SetZero(true)
    on *nnet_to_update) it will compute the gradient on this data.
    If nnet_to_update_ == NULL, no backpropagation is done.
    
    Note: we ignore any existing acoustic score in the lattice of "eg".
    
    For display purposes you should normalize the sum of this return value by
    dividing by the sum over the examples, of the number of frames
    (lat_.size()) times the weight.

    Something you need to be careful with is that the occupation counts and the
    derivative are, following tradition, missing a factor equal to the acoustic
    scale.  So you need to multiply them by that scale if you plan to do
    something like L-BFGS in which you look at both the derivatives and function
    values.  */
 
SignedLogDouble NnetDiscriminativeUnsupervisedUpdate(const AmNnet &am_nnet,
                                          const TransitionModel &tmodel,
                                          const NnetDiscriminativeUnsupervisedUpdateOptions &opts,
                                          const DiscriminativeUnsupervisedNnetExample &eg,
                                          Nnet *nnet_to_update,
                                          NnetDiscriminativeUnsupervisedStats *stats);

SignedLogDouble ComputeNnetDiscriminativeUnsupervisdObjf(const AmNnet &am_nnet,
                                          const TransitionModel &tmodel,
                                          const NnetDiscriminativeUnsupervisedUpdateOptions &opts,
                                          const DiscriminativeUnsupervisedNnetExample &eg);

} // namespace nnet2
} // namespace kaldi

#endif // KALDI_NNET2_NNET_COMPUTE_DISCRIMINATIVE_UNSUPERVISED_H_

