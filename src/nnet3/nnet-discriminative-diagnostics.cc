// nnet3/nnet-discriminative-diagnostics.cc

// Copyright  2012-2015    Johns Hopkins University (author: Daniel Povey)
// Copyright  2014-2015    Vimal Manohar

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

#include "nnet3/nnet-discriminative-diagnostics.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/discriminative-training.h"

namespace kaldi {
namespace nnet3 {

NnetDiscriminativeComputeObjf::NnetDiscriminativeComputeObjf(
    const NnetComputeProbOptions &nnet_config,
    const discriminative::DiscriminativeTrainingOptions &discriminative_training_config,
    const TransitionModel &tmodel,
    const VectorBase<BaseFloat> &priors,
    const Nnet &nnet):
    nnet_config_(nnet_config),
    discriminative_training_config_(discriminative_training_config),
    tmodel_(tmodel),
    log_priors_(priors),
    nnet_(nnet),
    compiler_(nnet, nnet_config_.optimize_config),
    deriv_nnet_(NULL),
    num_minibatches_processed_(0) {
  log_priors_.ApplyLog();
  if (nnet_config_.compute_deriv) {
    deriv_nnet_ = new Nnet(nnet_);
    bool is_gradient = true;  // force simple update
    SetZero(is_gradient, deriv_nnet_);
  }
}

const Nnet& NnetDiscriminativeComputeObjf::GetDeriv() const {
  if (deriv_nnet_ == NULL)
    KALDI_ERR << "GetDeriv() called when no derivatives were requested.";
  return *deriv_nnet_;
}

NnetDiscriminativeComputeObjf::~NnetDiscriminativeComputeObjf() {
  delete deriv_nnet_;  // delete does nothing if pointer is NULL.
}

void NnetDiscriminativeComputeObjf::Reset() {
  num_minibatches_processed_ = 0;
  objf_info_.clear();
  if (deriv_nnet_) {
    bool is_gradient = true;
    SetZero(is_gradient, deriv_nnet_);
  }
}

void NnetDiscriminativeComputeObjf::Compute(const NnetDiscriminativeExample &eg) {
  bool need_model_derivative = nnet_config_.compute_deriv,
      store_component_stats = false;
  ComputationRequest request;
  GetDiscriminativeComputationRequest(nnet_, eg, 
                                      need_model_derivative,
                                      store_component_stats,
                                      &request);
  const NnetComputation *computation = compiler_.Compile(request);
  NnetComputer computer(nnet_config_.compute_config, *computation,
                        nnet_, deriv_nnet_);
  // give the inputs to the computer object.
  computer.AcceptInputs(nnet_, eg.inputs);
  computer.Forward();
  this->ProcessOutputs(eg, &computer);
  if (nnet_config_.compute_deriv)
    computer.Backward();
}

void NnetDiscriminativeComputeObjf::ProcessOutputs(const NnetDiscriminativeExample &eg,
                                         NnetComputer *computer) {
  // There will normally be just one output here, named 'output',
  // but the code is more general than this.
  std::vector<NnetDiscriminativeSupervision>::const_iterator iter = eg.outputs.begin(),
      end = eg.outputs.end();
  for (; iter != end; ++iter) {
    const NnetDiscriminativeSupervision &sup = *iter;
    int32 node_index = nnet_.GetNodeIndex(sup.name);
    if (node_index < 0 ||
        !nnet_.IsOutputNode(node_index))
      KALDI_ERR << "Network has no output named " << sup.name;

    const CuMatrixBase<BaseFloat> &nnet_output = computer->GetOutput(sup.name);
    CuMatrix<BaseFloat> nnet_output_deriv;
    if (nnet_config_.compute_deriv)
      nnet_output_deriv.Resize(nnet_output.NumRows(), nnet_output.NumCols(),
                               kUndefined);

    discriminative::DiscriminativeTrainingStats stats;

    discriminative::ComputeDiscriminativeObjfAndDeriv(discriminative_training_config_, 
                                                      tmodel_, log_priors_,
                                                      sup.supervision, nnet_output,
                                                      &stats,
                                                      (nnet_config_.compute_deriv ?
                                                       &nnet_output_deriv : NULL));

    // note: in this context we don't want to apply 'sup.deriv_weights' because
    // this code is used only in combination, where it's part of an L-BFGS
    // optimization algorithm, and in that case if there is a mismatch between
    // the computed objective function and the derivatives, it may cause errors
    // in the optimization procedure such as early termination.  (line search
    // and conjugate gradient descent both rely on the derivatives being
    // accurate, and don't fail gracefully if the derivatives are not accurate).

    SimpleObjectiveInfo &totals = objf_info_[sup.name];
    totals.tot_weight += stats.tot_t_weighted;
    totals.tot_objective += stats.tot_objf;

    if (nnet_config_.compute_deriv)
      computer->AcceptOutputDeriv(sup.name, &nnet_output_deriv);

    num_minibatches_processed_++;
  }
}

bool NnetDiscriminativeComputeObjf::PrintTotalStats() const {
  bool ans = false;
  unordered_map<std::string, SimpleObjectiveInfo, StringHasher>::const_iterator
      iter, end;
  iter = objf_info_.begin();
  end = objf_info_.end();
  for (; iter != end; ++iter) {
    const std::string &name = iter->first;
    int32 node_index = nnet_.GetNodeIndex(name);
    KALDI_ASSERT(node_index >= 0);
    const SimpleObjectiveInfo &info = iter->second;
    KALDI_LOG << "Overall " << discriminative_training_config_.criterion
              << " objective for '"
              << name << "' is "
              << (info.tot_objective / info.tot_weight) << " per frame"
              << ", over " << info.tot_weight << " frames.";
    if (info.tot_weight > 0)
      ans = true;
  }
  return ans;
}


const SimpleObjectiveInfo* NnetDiscriminativeComputeObjf::GetObjective(
    const std::string &output_name) const {
  unordered_map<std::string, SimpleObjectiveInfo, StringHasher>::const_iterator
      iter = objf_info_.find(output_name);
  if (iter != objf_info_.end())
    return &(iter->second);
  else
    return NULL;
}

} // namespace nnet3
} // namespace kaldi

