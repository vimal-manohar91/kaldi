// chain/chain-training.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)


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


#ifndef KALDI_CHAIN_CHAIN_TRAINING_H_
#define KALDI_CHAIN_CHAIN_TRAINING_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "hmm/transition-model.h"
#include "chain/chain-den-graph.h"
#include "chain/chain-supervision.h"

namespace kaldi {
namespace chain {


struct ChainTrainingOptions {
  // l2 regularization constant on the 'chain' output; the actual term added to
  // the objf will be -0.5 times this constant times the squared l2 norm.
  // (squared so it's additive across the dimensions).  e.g. try 0.0005.
  BaseFloat l2_regularize;

  // Coefficient for 'leaky hmm'.  This means we have an epsilon-transition from
  // each state to a special state with probability one, and then another
  // epsilon-transition from that special state to each state, with probability
  // leaky_hmm_coefficient times [initial-prob of destination state].  Imagine
  // we make two copies of each state prior to doing this, version A and version
  // B, with transition from A to B, so we don't have to consider epsilon loops-
  // or just imagine the coefficient is small enough that we can ignore the
  // epsilon loops.
  // Note: we generally set leaky_hmm_coefficient to 0.1.
  BaseFloat leaky_hmm_coefficient;

  // Cross-entropy regularization constant.  (e.g. try 0.1).  If nonzero,
  // the network is expected to have an output named 'output-xent', which
  // should have a softmax as its final nonlinearity.
  BaseFloat xent_regularize;

  bool use_smbr_objective;
  bool exclude_silence;
  bool one_silence_class;

  std::string silence_pdfs_str;

  BaseFloat mmi_factor;
  BaseFloat ml_factor;
  BaseFloat kl_factor;
  BaseFloat smbr_factor;
  BaseFloat smbr_threshold;

  bool self_kl;
  bool norm_regularize;

  BaseFloat smbr_leaky_hmm_coefficient;
  bool smbr_use_numerator_post_targets;

  std::string smbr_factors_str, mmi_factors_str, ml_factors_str, kl_factors_str;

  ChainTrainingOptions(): l2_regularize(0.0), leaky_hmm_coefficient(1.0e-05),
                          xent_regularize(0.0), use_smbr_objective(false),
                          exclude_silence(false), one_silence_class(false),
                          mmi_factor(1.0), ml_factor(0.0), kl_factor(0.0),
                          smbr_factor(0.0), smbr_threshold(0.0), self_kl(false),
                          norm_regularize(false), 
                          smbr_leaky_hmm_coefficient(-1),
                          smbr_use_numerator_post_targets(false) { }

  void Register(OptionsItf *opts) {
    opts->Register("l2-regularize", &l2_regularize, "l2 regularization "
                   "constant for 'chain' training, applied to the output "
                   "of the neural net.");
    opts->Register("norm-regularize", &norm_regularize, 
                   "If true, then use l1 regularization on exponential of the "
                   "output of the neural net. Tends to make the "
                   "exp(output) small and more like probabilities.");
    opts->Register("leaky-hmm-coefficient", &leaky_hmm_coefficient, "Coefficient "
                   "that allows transitions from each HMM state to each other "
                   "HMM state, to ensure gradual forgetting of context (can "
                   "improve generalization).  For numerical reasons, may not be "
                   "exactly zero.");
    opts->Register("xent-regularize", &xent_regularize, "Cross-entropy "
                   "regularization constant for 'chain' training.  If "
                   "nonzero, the network is expected to have an output "
                   "named 'output-xent', which should have a softmax as "
                   "its final nonlinearity.");
    opts->Register("use-smbr-objective", &use_smbr_objective, 
                   "Use SMBR objective instead of MMI");
    opts->Register("silence-pdfs", &silence_pdfs_str,
                   "A comma-separated list of silence pdfs. "
                   "It makes sense only when the silence pdfs are "
                   "context-independent.");
    opts->Register("mmi-factor", &mmi_factor,
                   "When using smbr objective, interpolate mmi objective "
                   "with this weight");
    opts->Register("ml-factor", &ml_factor,
                   "When using smbr objective, interpolate ml objective "
                   "with this weight");
    opts->Register("smbr-factor", &smbr_factor,
                   "When using smbr objective, interpolate smbr objective "
                   "with this weight");
    opts->Register("exclude-silence", &exclude_silence,
                   "Exclude numerator posteriors "
                   "of silence pdfs from accuracy computation in "
                   "sMBR training. --silence-pdfs is required if "
                   "this option is true.");
    opts->Register("one-silence-class", &one_silence_class,
                   "Treat all silence pdfs as a single class for accuracy "
                   "computation in smBR training. --silence-pdfs is required "
                   "if this options is true.");
    opts->Register("smbr-threshold", &smbr_threshold,
                   "Posterior below this value is considered 0");
    opts->Register("smbr-factors", &smbr_factors_str,
                   "SMBR factors for each output");
    opts->Register("mmi-factors", &mmi_factors_str,
                   "MMI factors for each output");
    opts->Register("ml-factors", &ml_factors_str,
                   "ML factors for each output");
    opts->Register("kl-factors", &kl_factors_str,
                   "KL factors for each output");
    opts->Register("smbr-leaky-hmm-coefficient", &smbr_leaky_hmm_coefficient,
                   "leaky-hmm-coefficient for LF-sMBR training. If not "
                   "provided, will use --leaky-hmm-coefficient instead.");
    opts->Register("smbr-use-numerator-post-targets",
                   &smbr_use_numerator_post_targets,
                   "Use numerator posterior targets for computing "
                   "SMBR per-frame accuracies.");
  }
};


/**
   This function does both the numerator and denominator parts of the 'chain'
   computation in one call.

   @param [in] opts        Struct containing options
   @param [in] den_graph   The denominator graph, derived from denominator fst.
   @param [in] supervision  The supervision object, containing the supervision
                            paths and constraints on the alignment as an FST
   @param [in] nnet_output  The output of the neural net; dimension must equal
                          ((supervision.num_sequences * supervision.frames_per_sequence) by
                            den_graph.NumPdfs()).  The rows are ordered as: all sequences
                            for frame 0; all sequences for frame 1; etc.
   @param [out] objf       The [num - den] objective function computed for this
                           example; you'll want to divide it by 'tot_weight' before
                           displaying it.
   @param [out] l2_term  The l2 regularization term in the objective function, if
                           the --l2-regularize option is used.  To be added to 'o
   @param [out] weight     The weight to normalize the objective function by;
                           equals supervision.weight * supervision.num_sequences *
                           supervision.frames_per_sequence.
   @param [out] nnet_output_deriv  The derivative of the objective function w.r.t.
                           the neural-net output.  Only written to if non-NULL.
                           You don't have to zero this before passing to this function,
                           we zero it internally.
   @param [out] xent_output_deriv  If non-NULL, then the numerator part of the derivative
                           (which equals a posterior from the numerator
                           forward-backward, scaled by the supervision weight)
                           is written to here (this function will set it to the
                           correct size first; doing it this way reduces the
                           peak memory use).  xent_output_deriv will be used in
                           the cross-entropy regularization code; it is also
                           used in computing the cross-entropy objective value.
*/
void ComputeChainObjfAndDeriv(const ChainTrainingOptions &opts,
                              const DenominatorGraph &den_graph,
                              const Supervision &supervision,
                              const CuMatrixBase<BaseFloat> &nnet_output,
                              BaseFloat *objf,
                              BaseFloat *l2_term,
                              BaseFloat *weight,
                              CuMatrixBase<BaseFloat> *nnet_output_deriv,
                              CuMatrix<BaseFloat> *xent_output_deriv = NULL);

void ComputeChainNumeratorPost(const Supervision &supervision,
                               const CuMatrixBase<BaseFloat> &nnet_output,
                               CuMatrixBase<BaseFloat> *numerator_post);

/**
   This function does both the numerator and denominator parts of the 'chain'
   smbr computation in one call.

   @param [in] opts        Struct containing options
   @param [in] den_graph   The denominator graph, derived from denominator fst.
   @param [in] supervision  The supervision object, containing the supervision
                            paths and constraints on the alignment as an FST
   @param [in] nnet_output  The output of the neural net; dimension must equal
                          ((supervision.num_sequences * supervision.frames_per_sequence) by
                            den_graph.NumPdfs()).  The rows are ordered as: all sequences
                            for frame 0; all sequences for frame 1; etc.
   @param [out] objf       The smbr objective function computed for this
                           example; you'll want to divide it by 'tot_weight' before
                           displaying it.
   @param [out] l2_term  The l2 regularization term in the objective function, if
                           the --l2-regularize option is used.  To be added to 'o
   @param [out] weight     The weight to normalize the objective function by;
                           equals supervision.weight * supervision.num_sequences *
                           supervision.frames_per_sequence.
   @param [out] nnet_output_deriv  The derivative of the objective function w.r.t.
                           the neural-net output.  Only written to if non-NULL.
                           You don't have to zero this before passing to this function,
                           we zero it internally.
   @param [out] xent_output_deriv  If non-NULL, then the numerator part of the derivative
                           (which equals a posterior from the numerator forward-backward,
                           scaled by the supervision weight) is written to here.  This will
                           be used in the cross-entropy regularization code.  This value
                           is also used in computing the cross-entropy objective value.
*/
void ComputeChainSmbrObjfAndDeriv(
    const ChainTrainingOptions &opts,
    const DenominatorGraph &den_graph,
    const Supervision &supervision,
    const CuMatrixBase<BaseFloat> &nnet_output,
    BaseFloat *objf, BaseFloat *mmi_objf,
    BaseFloat *l2_term,
    BaseFloat *weight,
    CuMatrixBase<BaseFloat> *nnet_output_deriv,
    CuMatrix<BaseFloat> *xent_output_deriv = NULL,
    const CuArray<MatrixIndexT> *sil_indices = NULL);

/**
  This function uses supervision as numerator and does denominator computation.
  It can be uses, where numerator is fixed e.g. TS learning.
*/
void ComputeChainDenominatorObjfAndDeriv(const ChainTrainingOptions &opts,
                                    const DenominatorGraph &den_graph,
                                    const CuMatrixBase<BaseFloat> &nnet_output,
                                    BaseFloat supervision_weight, int32 num_sequences,
                                    BaseFloat *objf,
                                    BaseFloat *weight,
                                    CuMatrixBase<BaseFloat> *nnet_output_deriv,
                                    CuMatrixBase<BaseFloat> *xent_output_deriv = NULL);

}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_TRAINING_H_
