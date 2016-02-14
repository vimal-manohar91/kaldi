// nnet3/discriminative-training.h

// Copyright      2012-2015    Johns Hopkins University (author: Daniel Povey)
// Copyright      2014-2015    Vimal Manohar


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


#ifndef KALDI_NNET3_DISCRIMINATIVE_TRAINING_H_
#define KALDI_NNET3_DISCRIMINATIVE_TRAINING_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "hmm/transition-model.h"
#include "nnet3/discriminative-supervision.h"
#include "lat/lattice-functions.h"
#include "cudamatrix/cu-matrix-lib.h"

namespace kaldi {
namespace discriminative {

struct DiscriminativeTrainingOptions {
  std::string criterion; // "mmi" or "mpfe" or "smbr" or "nce" or "empfe" or "esmbr"
                         // If the criterion does not match the supervision
                         // object, the derivatives may not be very accurate
  BaseFloat acoustic_scale; // e.g. 0.1
  bool drop_frames; // for MMI, true if we ignore frames where alignment
                    // pdf-id is not in the lattice.
  bool one_silence_class;  // Affects MPFE/SMBR/EMPFE/ESMBR
  BaseFloat deletion_penalty;     // e.g. 0.1. Affects ESMBR and EMPFE.
  BaseFloat boost; // for MMI, boosting factor (would be Boosted MMI)... e.g. 0.1.
  
  std::string silence_phones_str; // colon-separated list of integer ids of silence phones,
                                  // for MPFE/SMBR/EMPFE/ESMBR only.

  BaseFloat weight_threshold; // e.g. 0.0
  BaseFloat xent_regularize;
  BaseFloat l2_regularize;
  bool debug_training;
  bool debug_training_advanced;

  DiscriminativeTrainingOptions(): criterion("smbr"), acoustic_scale(0.1),
                                   drop_frames(false),
                                   one_silence_class(false),
                                   deletion_penalty(0.0),
                                   boost(0.0), weight_threshold(0.0),
                                   xent_regularize(0.0), l2_regularize(0.0),
                                   debug_training(false), 
                                   debug_training_advanced(false) { }

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
    opts->Register("l2-regularize", &l2_regularize, "l2 regularization "
                   "constant for 'chain' training, applied to the output "
                   "of the neural net.");
    opts->Register("xent-regularize", &xent_regularize, "Cross-entropy "
                   "regularization constant for 'chain' training.  If "
                   "nonzero, the network is expected to have an output "
                   "named 'output-xent', which should have a softmax as "
                   "its final nonlinearity.");
    opts->Register("debug-training", &debug_training,
                   "Debug training using oracle alignment. Gives error "
                   "if oracle alignment is not found in examples");
    opts->Register("debug-training-advanced", &debug_training_advanced,
                   "Debug training using oracle alignment with all objective functions. "
                   "Gives error if oracle alignment is not found in examples");
  }
};

struct DiscriminativeTrainingStatsOptions {
  bool accumulate_gradients;
  bool accumulate_output;
  bool accumulate_counts;
  int32 num_pdfs;

  void Register(OptionsItf *opts) {
    opts->Register("accumulate-gradients", &accumulate_gradients,
                   "Accumulate gradients for debugging discriminative training");
    opts->Register("accumulate-counts", &accumulate_counts,
                   "Accumulate indicator counts of denominator pdfs " 
                   "for debugging discriminative training");
    opts->Register("accumulate-output", &accumulate_output,
                   "Accumulate nnet output "
                   "for debugging discriminative training");
    opts->Register("num-pdfs", &num_pdfs,
                   "Number of pdfs");
  }

  DiscriminativeTrainingStatsOptions() :
    accumulate_gradients(false), accumulate_output(false),
    accumulate_counts(false), num_pdfs(0) { }
};

struct DiscriminativeTrainingStats {
  double tot_t;         // total number of frames
  double tot_t_weighted; // total number of frames times weight.
  double tot_objf;      // for MMI, the (weighted) denominator likelihood; for
                        // everything else, the objective function.
  double tot_gradients; // for NCE, the gradients, for everything else 0
  double tot_num_count; // total count of numerator posterior for everything but NCE
  double tot_den_count; // total count of denominator posterior for everything but NCE
  double tot_num_objf;  // for MMI, the (weighted) numerator likelihood; for
                        // everything else 0

  DiscriminativeTrainingStatsOptions config;

  CuVector<double> gradients;
  CuVector<double> output;

  DiscriminativeTrainingStats(int32 num_pdfs) {
    config.accumulate_gradients = false; 
    config.accumulate_output = false; 
    config.accumulate_counts = false;
    config.num_pdfs = num_pdfs; 
    gradients.Resize(num_pdfs); 
    output.Resize(num_pdfs);
  }

  DiscriminativeTrainingStats() {
    std::memset(this, 0, sizeof(*this));
    config.accumulate_gradients = false; 
    config.accumulate_output = false; 
    config.accumulate_counts = false;
    config.num_pdfs = 0; 
  }

  DiscriminativeTrainingStats(DiscriminativeTrainingStatsOptions opts) : config(opts) { 
    gradients.Resize(opts.num_pdfs); 
    output.Resize(opts.num_pdfs);
  }
  
  void Reset() {
    gradients.SetZero();
    output.SetZero();
    
    tot_t = 0.0;
    tot_t_weighted = 0.0;
    tot_objf = 0.0;
    tot_gradients = 0.0;
    tot_num_count = 0.0;
    tot_den_count = 0.0;
    tot_num_objf = 0.0;
  }
  
  void SetConfig(const DiscriminativeTrainingStatsOptions &opts) {
    config = opts;
    gradients.Resize(opts.num_pdfs); 
    output.Resize(opts.num_pdfs);
  }

  void Print(const std::string &criterion, 
             bool print_avg_gradients = false, 
             bool print_avg_output = false,
             bool print_avg_counts = false) const;

  void PrintAll(const std::string &criterion) const {
    Print(criterion, true, true, true);
  }

  void PrintAvgGradientForPdf(int32 pdf_id) const;
  void Add(const DiscriminativeTrainingStats &other);

  inline double TotalObjf(const std::string &criterion) const {
    if (criterion == "mmi") return (tot_num_objf - tot_objf);
    return tot_objf;
  }

  inline double TotalT() const {
    return tot_t_weighted;
  }

  inline bool AccumulateGradients() const {
    return config.accumulate_gradients && gradients.Dim() > 0;
  }

  inline bool AccumulateOutput() const {
    return config.accumulate_output && output.Dim() > 0;
  }
};

/**
   This function does forward-backward on the numerator and denominator 
   lattices and computes derivates wrt to the output for the specified 
   objective function.

   @param [in] opts        Struct containing options
   @param [in] supervision  The supervision object, containing the numerator
                            and denominator paths. The denominator is 
                            always a lattice. The numerator can either be 
                            a lattice or an alignment.
   @param [in] nnet_output  The output of the neural net; dimension must equal
                          ((supervision.num_sequences * supervision.frames_per_sequence) by
                            den_graph.NumPdfs()).
   @param [out] objf       The objective function computed for this
                           example; you'll want to divide it by 'tot_weight' before
                           displaying it.
   @param [out] weight     The weight to normalize the objective function by;
                           equals supervision.weight * supervision.num_sequences *
                           supervision.frames_per_sequence.
   @param [out] nnet_output_deriv  The derivative of the objective function w.r.t.
                           the neural-net output.  Only written to if non-NULL.
                           You don't have to zero this before passing to this function,
                           we zero it internally.
*/
void ComputeDiscriminativeObjfAndDeriv(const DiscriminativeTrainingOptions &opts,
                                       const TransitionModel &tmodel,
                                       const CuVectorBase<BaseFloat> &log_priors,
                                       const DiscriminativeSupervision &supervision,
                                       const CuMatrixBase<BaseFloat> &nnet_output,
                                       DiscriminativeTrainingStats *stats,
                                       BaseFloat *l2_term,
                                       CuMatrixBase<BaseFloat> *nnet_output_deriv,
                                       CuMatrixBase<BaseFloat> *xent_output_deriv);

}  // namespace discriminative
}  // namespace kaldi

#endif  // KALDI_NNET3_DISCRIMINATIVE_TRAINING_H_


