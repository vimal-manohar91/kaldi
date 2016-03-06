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
                                   const VectorBase<BaseFloat> &priors,
                                   Nnet *nnet):
    opts_(opts), tmodel_(tmodel), log_priors_(priors),
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
  log_priors_.ApplyLog();
}


void NnetDiscriminativeTrainer::Train(const NnetDiscriminativeExample &eg) {
  bool need_model_derivative = true;
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  bool use_xent_regularization = (opts_.discriminative_training_config.xent_regularize != 0.0);
  ComputationRequest request;
  GetDiscriminativeComputationRequest(*nnet_, eg, need_model_derivative,
                                      nnet_config.store_component_stats,
                                      use_xent_regularization,
                                      need_model_derivative,
                                      &request);
  const NnetComputation *computation = compiler_.Compile(request);

  NnetComputer computer(nnet_config.compute_config, *computation,
                        *nnet_,
                        (delta_nnet_ == NULL ? nnet_ : delta_nnet_));
  // give the inputs to the computer object.
  computer.AcceptInputs(*nnet_, eg.inputs);
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
    
    bool use_xent = (opts_.discriminative_training_config.xent_regularize != 0.0);
    std::string xent_name = sup.name + "-xent";  // typically "output-xent".
    CuMatrix<BaseFloat> xent_deriv;
    if (use_xent)
      xent_deriv.Resize(nnet_output.NumRows(), nnet_output.NumCols(),
                               kUndefined);

    discriminative::DiscriminativeTrainingStats stats(opts_.discriminative_training_stats_config);
    
    if (objf_info_.count(sup.name) == 0)
      objf_info_[sup.name].stats.SetConfig(opts_.discriminative_training_stats_config);

    BaseFloat tot_l2_term = 0.0;

    ComputeDiscriminativeObjfAndDeriv(opts_.discriminative_training_config, 
                                      tmodel_, log_priors_,
                                      sup.supervision, nnet_output,
                                      &stats, &tot_l2_term,
                                      &nnet_output_deriv,
                                      (use_xent ? &xent_deriv : NULL));
    
    if (use_xent) {
      // this block computes the cross-entropy objective.
      const CuMatrixBase<BaseFloat> &xent_output = computer->GetOutput(xent_name);
      // at this point, xent_deriv is posteriors derived from the numerator
      // computation.  note, xent_objf has a factor of '.supervision.weight'
      BaseFloat xent_objf = TraceMatMat(xent_output, xent_deriv, kTrans);
      if (xent_objf != xent_objf) {
        BaseFloat default_objf = -10;
        xent_objf = default_objf;
      }

      objf_info_[xent_name].UpdateStats(xent_name, "xent",
                                        opts_.nnet_config.print_interval,
                                        num_minibatches_processed_,
                                        stats.TotalT(), xent_objf);
    }

    if (opts_.apply_deriv_weights && sup.deriv_weights.Dim() != 0) {
      CuVector<BaseFloat> cu_deriv_weights(sup.deriv_weights);
      nnet_output_deriv.MulRowsVec(cu_deriv_weights);
      if (use_xent)
        xent_deriv.MulRowsVec(cu_deriv_weights);
    }

    computer->AcceptOutputDeriv(sup.name, &nnet_output_deriv);

    objf_info_[sup.name].UpdateStats(sup.name, opts_.discriminative_training_config.criterion,
                                     opts_.nnet_config.print_interval,
                                     num_minibatches_processed_++,
                                     stats);
    
    if (use_xent) {
      xent_deriv.Scale(opts_.discriminative_training_config.xent_regularize);
      computer->AcceptOutputDeriv(xent_name, &xent_deriv);
    }
  }
}


bool NnetDiscriminativeTrainer::PrintTotalStats() const {
  unordered_map<std::string, DiscriminativeObjectiveFunctionInfo>::const_iterator
      iter = objf_info_.begin(),
      end = objf_info_.end();
  bool ans = false;
  for (; iter != end; ++iter) {
    const std::string &name = iter->first;
    const DiscriminativeObjectiveFunctionInfo &info = iter->second;
    bool ret = info.PrintTotalStats(name, opts_.discriminative_training_config.criterion);
    ans = ans || ret;
  }

  return ans;
}


void DiscriminativeObjectiveFunctionInfo::UpdateStats(
    const std::string &output_name,
    const std::string &criterion,
    int32 minibatches_per_phase,
    int32 minibatch_counter,
    BaseFloat this_minibatch_weight,
    BaseFloat this_minibatch_tot_objf,
    BaseFloat this_minibatch_tot_aux_objf) {
  int32 phase = minibatch_counter / minibatches_per_phase;
  if (phase != current_phase) {
    KALDI_ASSERT(phase == current_phase + 1); // or doesn't really make sense.
    PrintStatsForThisPhase(output_name, criterion, minibatches_per_phase);
    current_phase = phase;
    stats_this_phase.Reset();
    tot_aux_objf_this_phase = 0.0;
  }
  stats_this_phase.tot_t_weighted += this_minibatch_weight;
  stats_this_phase.tot_objf += this_minibatch_tot_objf;
  tot_aux_objf_this_phase += this_minibatch_tot_aux_objf;

  stats.Add(stats_this_phase);
  tot_aux_objf += this_minibatch_tot_aux_objf;
}

void DiscriminativeObjectiveFunctionInfo::UpdateStats(
    const std::string &output_name,
    const std::string &criterion,
    int32 minibatches_per_phase,
    int32 minibatch_counter,
    discriminative::DiscriminativeTrainingStats this_minibatch_stats,
    BaseFloat this_minibatch_tot_aux_objf) {
  int32 phase = minibatch_counter / minibatches_per_phase;
  if (phase != current_phase) {
    KALDI_ASSERT(phase == current_phase + 1); // or doesn't really make sense.
    PrintStatsForThisPhase(output_name, criterion, minibatches_per_phase);
    current_phase = phase;
    stats_this_phase.Reset();
    tot_aux_objf_this_phase = 0.0;
  }
  stats_this_phase.Add(this_minibatch_stats);
  tot_aux_objf_this_phase += this_minibatch_tot_aux_objf;

  stats.Add(stats_this_phase);
  tot_aux_objf += this_minibatch_tot_aux_objf;
}
void DiscriminativeObjectiveFunctionInfo::PrintStatsForThisPhase(
    const std::string &output_name,
    const std::string &criterion,
    int32 minibatches_per_phase) const {
  int32 start_minibatch = current_phase * minibatches_per_phase,
      end_minibatch = start_minibatch + minibatches_per_phase - 1;

  if (tot_aux_objf_this_phase == 0.0) {
    KALDI_LOG << "Average objective function for '" << output_name
              << "' for minibatches " << start_minibatch
              << '-' << end_minibatch << " is "
              << (stats_this_phase.TotalObjf(criterion) / stats_this_phase.TotalT()) << " over "
              << stats_this_phase.TotalT() << " frames.";
  } else {
    BaseFloat objf = (stats_this_phase.TotalObjf(criterion) / stats_this_phase.TotalT()),
        aux_objf = (tot_aux_objf_this_phase / stats_this_phase.TotalT());
    KALDI_LOG << "Average objective function for '" << output_name
              << "' for minibatches " << start_minibatch
              << '-' << end_minibatch << " is "
              << objf << " + " << aux_objf << " = " 
              << " over " << stats_this_phase.TotalT() << " frames.";
  }
}

bool DiscriminativeObjectiveFunctionInfo::PrintTotalStats(const std::string &name,
                const std::string &criterion) const {
  BaseFloat objf = stats.TotalObjf(criterion) /stats.TotalT(),
        aux_objf = (tot_aux_objf / stats.TotalT());
  if (tot_aux_objf == 0.0) {
    KALDI_LOG << "Overall average objective function for '" << name << "' is "
              << objf << " over " << stats.TotalT() << " frames.";
  } else {
    KALDI_LOG << "Overall average objective function for '" << name << "' is "
              << objf << " + " << aux_objf << " = " 
              << " over " << stats.TotalT() << " frames.";
  }
  KALDI_LOG << "[this line is to be parsed by a script:] "
            << criterion << "-per-frame="
            << objf;
  return (stats.TotalT() != 0.0);
}


NnetDiscriminativeTrainer::~NnetDiscriminativeTrainer() {
  delete delta_nnet_;
}


} // namespace nnet3
} // namespace kaldi

