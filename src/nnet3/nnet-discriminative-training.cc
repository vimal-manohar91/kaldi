// nnet3/nnet-discriminative-training.cc

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

#include "nnet3/nnet-discriminative-training.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetDiscriminativeTrainer::NnetDiscriminativeTrainer(
                                   const NnetDiscriminativeTrainingOptions &opts,
                                   const TransitionModel &tmodel,
                                   const Vector<BaseFloat> &priors,
                                   Nnet *nnet):
    opts_(opts), tmodel_(tmodel), priors_(priors),
    nnet_(nnet),
    compiler_(*nnet, opts_.nnet_config.optimize_config),
    num_minibatches_processed_(0) {
  if (opts.nnet_config.zero_component_stats)
    ZeroComponentStats(nnet);
  if (opts.nnet_config.momentum == 0.0 &&
      opts.nnet_config.max_param_change == 0.0) {
    delta_nnet_= NULL;
  } else {
    KALDI_ASSERT(opts.nnet_config.momentum >= 0.0 &&
                 opts.nnet_config.max_param_change >= 0.0);
    delta_nnet_ = nnet_->Copy();
    bool is_gradient = false;  // setting this to true would disable the
                               // natural-gradient updates.
    SetZero(is_gradient, delta_nnet_);
  }
}


void NnetDiscriminativeTrainer::Train(const NnetDiscriminativeExample &eg) {
  bool need_model_derivative = true;
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  ComputationRequest request;
  GetDiscriminativeComputationRequest(*nnet_, eg, need_model_derivative,
                                      nnet_config.store_component_stats,
                                      &request);
  const NnetComputation *computation = compiler_.Compile(request);

  NnetComputer computer(nnet_config.compute_config, *computation,
                        *nnet_,
                        (delta_nnet_ == NULL ? nnet_ : delta_nnet_));
  // give the inputs to the computer object.
  computer.AcceptInputs(*nnet_, chain_eg.inputs);
  computer.Forward();

  this->ProcessOutputs(eg, &computer);
  computer.Backward();

  if (delta_nnet_ != NULL) {
    BaseFloat scale = (1.0 - nnet_config.momentum);
    if (nnet_config.max_param_change != 0.0) {
      BaseFloat param_delta =
          std::sqrt(DotProduct(*delta_nnet_, *delta_nnet_)) * scale;
      if (param_delta > nnet_config.max_param_change) {
        if (param_delta - param_delta != 0.0) {
          KALDI_WARN << "Infinite parameter change, will not apply.";
          SetZero(false, delta_nnet_);
        } else {
          scale *= nnet_config.max_param_change / param_delta;
          KALDI_LOG << "Parameter change too big: " << param_delta << " > "
                    << "--max-param-change=" << nnet_config.max_param_change
                    << ", scaling by "
                    << nnet_config.max_param_change / param_delta;
        }
      }
    }
    AddNnet(*delta_nnet_, scale, nnet_);
    ScaleNnet(nnet_config.momentum, delta_nnet_);
  }
}


void NnetDiscriminativeTrainer::ProcessOutputs(const NnetDiscriminativeExample &eg,
                                               NnetComputer *computer) {
  // normally the eg will have just one output named 'output', but
  // we don't assume this.
  std::vector<NnetDiscriminativeSupervision>::const_iterator iter = eg.outputs.begin(),
      end = eg.outputs.end();
  for (; iter != end; ++iter) {
    const NnetDiscriminativeSupervision &sup = *iter;
    int32 node_index = nnet_->GetNodeIndex(sup.name);
    if (node_index < 0 ||
        !nnet_->IsOutputNode(node_index))
      KALDI_ERR << "Network has no output named " << sup.name;

    const CuMatrixBase<BaseFloat> &nnet_output = computer->GetOutput(sup.name);
    CuMatrix<BaseFloat> nnet_output_deriv(nnet_output.NumRows(),
                                          nnet_output.NumCols(),
                                          kUndefined);

    BaseFloat tot_objf, tot_weight;

    ComputeDiscriminativeObjfAndDeriv(opts_.chain_config, tmodel_, priors_,
                                      sup.supervision, nnet_output,
                                      &tot_objf, &tot_weight,
                                      &nnet_output_deriv);

    if (opts_.apply_deriv_weights && sup.deriv_weights.Dim() != 0) {
      CuVector<BaseFloat> cu_deriv_weights(sup.deriv_weights);
      nnet_output_deriv.MulRowsVec(cu_deriv_weights);
    }

    computer->AcceptOutputDeriv(sup.name, &nnet_output_deriv);

    objf_info_[sup.name].UpdateStats(sup.name, opts_.nnet_config.print_interval,
                                     num_minibatches_processed_++,
                                     tot_weight, tot_objf);
  }
}


bool NnetDiscriminativeTrainer::PrintTotalStats() const {
  unordered_map<std::string, ObjectiveFunctionInfo>::const_iterator
      iter = objf_info_.begin(),
      end = objf_info_.end();
  bool ans = false;
  for (; iter != end; ++iter) {
    const std::string &name = iter->first;
    const ObjectiveFunctionInfo &info = iter->second;
    ans = ans || info.PrintTotalStats(name);
  }
  return ans;
}


NnetDiscriminativeTrainer::~NnetDiscriminativeTrainer() {
  delete delta_nnet_;
}


} // namespace nnet3
} // namespace kaldi

