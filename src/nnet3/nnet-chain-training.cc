// nnet3/nnet-chain-training.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)
//                2016    Xiaohui Zhang

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

#include "nnet3/nnet-chain-training.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {


NnetChainTrainer::NnetChainTrainer(const NnetChainTrainingOptions &opts,
                                   const fst::StdVectorFst &den_fst,
                                   Nnet *nnet):
    opts_(opts),
    den_graph_(den_fst, nnet->OutputDim("output")),
    nnet_(nnet),
    compiler_(*nnet, opts_.nnet_config.optimize_config,
              opts_.nnet_config.compiler_config),
    num_minibatches_processed_(0),
    srand_seed_(RandInt(0, 100000)) {
  if (opts.nnet_config.zero_component_stats)
    ZeroComponentStats(nnet);
  KALDI_ASSERT(opts.nnet_config.momentum >= 0.0 &&
               opts.nnet_config.max_param_change >= 0.0 &&
               opts.nnet_config.backstitch_training_interval > 0);
  delta_nnet_ = nnet_->Copy();
  ScaleNnet(0.0, delta_nnet_);
  const int32 num_updatable = NumUpdatableComponents(*delta_nnet_);
  num_max_change_per_component_applied_.resize(num_updatable, 0);
  num_max_change_global_applied_ = 0;

  if (opts.nnet_config.read_cache != "") {
    bool binary;
    try {
      Input ki(opts.nnet_config.read_cache, &binary);
      compiler_.ReadCache(ki.Stream(), binary);
      KALDI_LOG << "Read computation cache from " << opts.nnet_config.read_cache;
    } catch (...) {
      KALDI_WARN << "Could not open cached computation. "
                    "Probably this is the first training iteration.";
    }
  }

  if (opts.chain_config.use_smbr_objective &&
      (opts.chain_config.exclude_silence || opts.chain_config.one_silence_class)) {
    if (opts.chain_config.silence_pdfs_str.empty()) {
      KALDI_ERR << "--silence-pdfs is required if --exclude-silence or "
                << "--one-silence-class is true.";
    }

    std::vector<std::string> silence_pdfs;
    SplitStringToVector(opts.chain_config.silence_pdfs_str, ":,", false, 
                        &silence_pdfs);

    int32 num_pdfs = nnet->OutputDim("output");
    std::vector<int32> indices(num_pdfs, -1);

    if (opts.chain_config.exclude_silence) {
      for (size_t i = 0; i < num_pdfs; i++) {
        indices[i] = i;
      }

      for (std::vector<std::string>::iterator it = silence_pdfs.begin();
           it != silence_pdfs.end(); ++it) {
        int32 pdf = std::atoi(it->c_str());
        if (pdf > num_pdfs) 
          KALDI_ERR << "Invalid pdf " << pdf << " in silence-pdfs "
                    << opts.chain_config.silence_pdfs_str;
        indices[pdf] = -1;
      }
    } else {
      for (std::vector<std::string>::iterator it = silence_pdfs.begin();
           it != silence_pdfs.end(); ++it) {
        int32 pdf = std::atoi(it->c_str());
        if (pdf > num_pdfs) 
          KALDI_ERR << "Invalid pdf " << pdf << " in silence-pdfs "
                    << opts.chain_config.silence_pdfs_str;
        indices[pdf] = pdf;
      }
    }

    sil_indices_.Resize(num_pdfs);
    sil_indices_.CopyFromVec(indices);
  }

  if (!opts.chain_config.smbr_factors_str.empty())
    ParseObjectiveScales(opts.chain_config.smbr_factors_str,
                         &smbr_factors_);
  if (!opts.chain_config.mmi_factors_str.empty())
    ParseObjectiveScales(opts.chain_config.mmi_factors_str,
                         &mmi_factors_);
  if (!opts.chain_config.ml_factors_str.empty())
    ParseObjectiveScales(opts.chain_config.ml_factors_str,
                         &ml_factors_);
}

void NnetChainTrainer::Train(const NnetChainExample &chain_eg) {
  bool need_model_derivative = true;
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  bool use_xent_regularization = (opts_.chain_config.xent_regularize != 0.0);
  ComputationRequest request;
  GetChainComputationRequest(*nnet_, chain_eg, need_model_derivative,
                             nnet_config.store_component_stats,
                             use_xent_regularization, need_model_derivative,
                             &request);
  const NnetComputation *computation = compiler_.Compile(request);

  if (nnet_config.backstitch_training_scale > 0.0 && num_minibatches_processed_
      % nnet_config.backstitch_training_interval ==
      srand_seed_ % nnet_config.backstitch_training_interval) {
    // backstitch training is incompatible with momentum > 0
    KALDI_ASSERT(nnet_config.momentum == 0.0);
    FreezeNaturalGradient(true, delta_nnet_);
    bool is_backstitch_step1 = true;
    srand(srand_seed_ + num_minibatches_processed_);
    ResetGenerators(nnet_);
    TrainInternalBackstitch(chain_eg, *computation, is_backstitch_step1);
    FreezeNaturalGradient(false, delta_nnet_); // un-freeze natural gradient
    is_backstitch_step1 = false;
    srand(srand_seed_ + num_minibatches_processed_);
    ResetGenerators(nnet_);
    TrainInternalBackstitch(chain_eg, *computation, is_backstitch_step1);
  } else { // conventional training
    TrainInternal(chain_eg, *computation);
  }

  num_minibatches_processed_++;
}

// This object exists to help avoid memory fragmentation: it allocates,
// but does not use, the exact sizes of memory that are going to be needed
// in ComputeChainObjfAndDeriv().
class ChainTrainerMemoryHolder {
 public:
  ChainTrainerMemoryHolder(const Nnet &nnet,
                           int32 num_den_graph_states,
                           const NnetChainExample &eg,
                           bool use_smbr_objective = false);
 private:
  CuMatrix<BaseFloat> nnet_output_deriv_;
  CuMatrix<BaseFloat> xent_output_deriv_;
  CuMatrix<BaseFloat> beta_;
  CuMatrix<BaseFloat> alpha_;
  CuMatrix<BaseFloat> beta_smbr_;
  CuMatrix<BaseFloat> alpha_smbr_;
};

ChainTrainerMemoryHolder::ChainTrainerMemoryHolder(const Nnet &nnet,
                                                   int32 den_graph_states,
                                                   const NnetChainExample &eg,
                                                   bool use_smbr_objective) {

  std::vector<NnetChainSupervision>::const_iterator iter = eg.outputs.begin(),
      end = eg.outputs.end();

  int32 max_rows = 0,
      max_cols = 0;

  size_t max_frames_per_sequence = 0,
         max_sequence_size = 0,
         max_alpha_matrix_size = 0;

  for (; iter != end; ++iter) {
    // there will normally be just one of these things; we'll normally loop once.
    const NnetChainSupervision &sup = *iter;

    int32 output_rows = sup.supervision.num_sequences * sup.supervision.frames_per_sequence;
    int32 output_cols = nnet.OutputDim("output");

    size_t curr_frames_per_sequence = output_rows / sup.supervision.num_sequences + 1;
    size_t den_graph_size = den_graph_states + 1;
    size_t curr_sequence_size = den_graph_size * sup.supervision.num_sequences;
    size_t curr_alpha_matrix_size = curr_frames_per_sequence * curr_sequence_size;

    if (curr_alpha_matrix_size > max_alpha_matrix_size) {
      max_alpha_matrix_size = curr_alpha_matrix_size;
      max_frames_per_sequence = curr_frames_per_sequence;
      max_sequence_size = curr_sequence_size;
    }

    size_t matrix_size = output_rows * output_cols;
    if (matrix_size > (max_rows * max_cols)) {
      max_rows = output_rows;
      max_cols = output_cols;
    }
  }

  // the sequence of resizes is in a specific order (bigger to smaller)
  // so that the cudaMalloc won't trash the memory it has already
  // alloc'd in the previous iterations
  alpha_.Resize(max_frames_per_sequence,
                max_sequence_size,
                kUndefined);

  nnet_output_deriv_.Resize(max_rows, max_cols, kUndefined);
  // note: the same block of memory can be used for xent_output_deriv_ as is
  // used for exp_nnet_output_transposed_ in chain-training.cc.
  xent_output_deriv_.Resize(max_rows, max_cols,
                            kUndefined, kStrideEqualNumCols);

  beta_.Resize(2, max_sequence_size, kUndefined);

  if (use_smbr_objective) {
    alpha_smbr_.Resize(max_frames_per_sequence,
                       max_sequence_size,
                       kUndefined);
    beta_smbr_.Resize(2, max_sequence_size, kUndefined);
  }
}

/*
ChainTrainerMemoryHolder::ChainTrainerMemoryHolder(const Nnet &nnet,
                                                   int32 den_graph_states,
                                                   const NnetExample &eg) {

  std::vector<NnetIo>::const_iterator iter = eg.io.begin(),
      end = eg.io.end();

  int32 max_rows = 0,
      max_cols = 0;

  size_t max_frames_per_sequence = 0,
         max_sequence_size = 0,
         max_alpha_matrix_size = 0;

  for (; iter != end; ++iter) {
    const NnetIo &io = *iter;
    int32 node_index = nnet.GetNodeIndex(io.name);
    KALDI_ASSERT(node_index >= 0);
    if (!nnet.IsOutputNode(node_index)) continue;

    int32 output_rows = io.features.NumRows();
    int32 output_cols = nnet.OutputDim("output");

    int32 num_sequences = NumSequencesInChainEg(io.indexes);
    size_t curr_frames_per_sequence = output_rows / num_sequences + 1;
    size_t den_graph_size = den_graph_states + 1;
    size_t curr_sequence_size = den_graph_size * num_sequences;
    size_t curr_alpha_matrix_size = curr_frames_per_sequence * curr_sequence_size;

    if (curr_alpha_matrix_size > max_alpha_matrix_size) {
      max_alpha_matrix_size = curr_alpha_matrix_size;
      max_frames_per_sequence = curr_frames_per_sequence;
      max_sequence_size = curr_sequence_size;
    }

    size_t matrix_size = output_rows * output_cols;
    if (matrix_size > (max_rows * max_cols)) {
      max_rows = output_rows;
      max_cols = output_cols;
    }
  }

  // the sequence of resizes is in a specific order (bigger to smaller)
  // so that the cudaMalloc won't trash the memory it has already
  // alloc'd in the previous iterations
  alpha_.Resize(max_frames_per_sequence,
                max_sequence_size,
                kUndefined);


  nnet_output_deriv_.Resize(max_rows, max_cols, kUndefined);
  // note: the same block of memory can be used for xent_output_deriv_ as is
  // used for exp_nnet_output_transposed_ in chain-training.cc.
  xent_output_deriv_.Resize(max_rows, max_cols,
                            kUndefined, kStrideEqualNumCols);

  beta_.Resize(2, max_sequence_size, kUndefined);
}

void NnetChainTrainer::TrainInternal(const NnetExample &eg,
                                     const NnetComputation &computation) {
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  // note: because we give the 1st arg (nnet_) as a pointer to the
  // constructor of 'computer', it will use that copy of the nnet to
  // store stats.
  NnetComputer computer(nnet_config.compute_config, computation,
                        nnet_, delta_nnet_);

  // reserve the memory needed in ProcessOutputs (before memory gets fragmented
  // by the call to computer.Run().
  ChainTrainerMemoryHolder *memory_holder =
      new ChainTrainerMemoryHolder(*nnet_, den_graph_.NumStates(), eg);

  // give the inputs to the computer object
  computer.AcceptInputs(*nnet_, eg.io);
  computer.Run();

  // 'this->ProcessOutputs()' is going to need the same sizes as are stored in
  // 'memory_holder'.
  delete memory_holder;

  this->ProcessOutputs(false, eg, &computer);
  computer.Run();

  // If relevant, add in the part of the gradient that comes from L2
  // regularization.
  ApplyL2Regularization(*nnet_,
                        GetNumNvalues(eg.io, false) *
                        nnet_config.l2_regularize_factor,
                        delta_nnet_);

  // Updates the parameters of nnet
  bool success = UpdateNnetWithMaxChange(*delta_nnet_,
      nnet_config.max_param_change, 1.0, 1.0 - nnet_config.momentum, nnet_,
      &num_max_change_per_component_applied_, &num_max_change_global_applied_);

  // Scale down the batchnorm stats (keeps them fresh... this affects what
  // happens when we use the model with batchnorm test-mode set).
  ScaleBatchnormStats(nnet_config.batchnorm_stats_scale, nnet_);

  // The following will only do something if we have a LinearComponent
  // or AffineComponent with orthonormal-constraint set to a nonzero value.
  ConstrainOrthonormal(nnet_);

  // Scale delta_nnet
  if (success)
    ScaleNnet(nnet_config.momentum, delta_nnet_);
  else
    ScaleNnet(0.0, delta_nnet_);
}
*/

void NnetChainTrainer::TrainInternal(const NnetChainExample &eg,
                                     const NnetComputation &computation) {
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  // note: because we give the 1st arg (nnet_) as a pointer to the
  // constructor of 'computer', it will use that copy of the nnet to
  // store stats.
  NnetComputer computer(nnet_config.compute_config, computation,
                        nnet_, delta_nnet_);

  // reserve the memory needed in ProcessOutputs (before memory gets fragmented
  // by the call to computer.Run().
  ChainTrainerMemoryHolder *memory_holder =
      new ChainTrainerMemoryHolder(*nnet_, den_graph_.NumStates(), eg,
                                   opts_.chain_config.use_smbr_objective);

  // give the inputs to the computer object.
  computer.AcceptInputs(*nnet_, eg.inputs);
  computer.Run();

  // 'this->ProcessOutputs()' is going to need the same sizes as are stored in
  // 'memory_holder'.
  delete memory_holder;

  // Probably could be merged in a single call PreallocateChainTrainerMemory(*nnet_, eg) ?
  this->ProcessOutputs(false, eg, &computer);
  computer.Run();

  // If relevant, add in the part of the gradient that comes from L2
  // regularization.
  ApplyL2Regularization(*nnet_,
                        GetNumNvalues(eg.inputs, false) *
                        nnet_config.l2_regularize_factor,
                        delta_nnet_);

  // Updates the parameters of nnet
  bool success = UpdateNnetWithMaxChange(*delta_nnet_,
      nnet_config.max_param_change, 1.0, 1.0 - nnet_config.momentum, nnet_,
      &num_max_change_per_component_applied_, &num_max_change_global_applied_);

  // Scale down the batchnorm stats (keeps them fresh... this affects what
  // happens when we use the model with batchnorm test-mode set).
  ScaleBatchnormStats(nnet_config.batchnorm_stats_scale, nnet_);

  // The following will only do something if we have a LinearComponent
  // or AffineComponent with orthonormal-constraint set to a nonzero value.
  ConstrainOrthonormal(nnet_);

  // Scale delta_nnet
  if (success)
    ScaleNnet(nnet_config.momentum, delta_nnet_);
  else
    ScaleNnet(0.0, delta_nnet_);
}

void NnetChainTrainer::TrainInternalBackstitch(const NnetChainExample &eg,
                                               const NnetComputation &computation,
                                               bool is_backstitch_step1) {
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  // note: because we give the 1st arg (nnet_) as a pointer to the
  // constructor of 'computer', it will use that copy of the nnet to
  // store stats.
  NnetComputer computer(nnet_config.compute_config, computation,
                        nnet_, delta_nnet_);
  // give the inputs to the computer object.
  computer.AcceptInputs(*nnet_, eg.inputs);
  computer.Run();

  bool is_backstitch_step2 = !is_backstitch_step1;
  this->ProcessOutputs(is_backstitch_step2, eg, &computer);
  computer.Run();

  BaseFloat max_change_scale, scale_adding;
  if (is_backstitch_step1) {
    // max-change is scaled by backstitch_training_scale;
    // delta_nnet is scaled by -backstitch_training_scale when added to nnet;
    max_change_scale = nnet_config.backstitch_training_scale;
    scale_adding = -nnet_config.backstitch_training_scale;
  } else {
    // max-change is scaled by 1 + backstitch_training_scale;
    // delta_nnet is scaled by 1 + backstitch_training_scale when added to nnet;
    max_change_scale = 1.0 + nnet_config.backstitch_training_scale;
    scale_adding = 1.0 + nnet_config.backstitch_training_scale;
  }

  // If relevant, add in the part of the gradient that comes from L2
  // regularization.  It may not be optimally inefficient to do it on both
  // passes of the backstitch, like we do here, but it probably minimizes
  // any harmful interactions with the max-change.
  ApplyL2Regularization(*nnet_,
                        scale_adding * GetNumNvalues(eg.inputs, false) *
                        nnet_config.l2_regularize_factor,
                        delta_nnet_);

  // Updates the parameters of nnet
  UpdateNnetWithMaxChange(*delta_nnet_,
      nnet_config.max_param_change, max_change_scale, scale_adding, nnet_,
      &num_max_change_per_component_applied_, &num_max_change_global_applied_);

  if (is_backstitch_step1) {
    // The following will only do something if we have a LinearComponent or
    // AffineComponent with orthonormal-constraint set to a nonzero value. We
    // choose to do this only on the 1st backstitch step, for efficiency.
    ConstrainOrthonormal(nnet_);
  }

  if (!is_backstitch_step1) {
    // Scale down the batchnorm stats (keeps them fresh... this affects what
    // happens when we use the model with batchnorm test-mode set).  Do this
    // after backstitch step 2 so that the stats are scaled down before we start
    // the next minibatch.
    ScaleBatchnormStats(nnet_config.batchnorm_stats_scale, nnet_);
  }

  ScaleNnet(0.0, delta_nnet_);
}

/*
void NnetChainTrainer::ProcessOutputs(bool is_backstitch_step2,
                                      const NnetExample &eg,
                                      NnetComputer *computer) {
  // In backstitch training, the output-name with the "_backstitch" suffix is
  // the one computed after the first, backward step of backstitch.
  const std::string suffix = (is_backstitch_step2 ? "_backstitch" : "");
  std::vector<NnetIo>::const_iterator iter = eg.io.begin(),
    end = eg.io.end();
  for (; iter != end; ++iter) {
    const NnetIo &io = *iter;
    int32 node_index = nnet_->GetNodeIndex(io.name);
    KALDI_ASSERT(node_index >= 0);
    if (nnet_->IsOutputNode(node_index)) {
      const CuMatrixBase<BaseFloat> &nnet_output = computer->GetOutput(io.name);
      CuMatrix<BaseFloat> nnet_output_deriv(nnet_output.NumRows(),
                                            nnet_output.NumCols(),
                                            kUndefined);
      bool use_xent = (opts_.chain_config.xent_regularize != 0.0);
      std::string xent_name = io.name + "-xent";  // typically "output-xent".
      CuMatrix<BaseFloat> xent_deriv;

      BaseFloat tot_objf, tot_l2_term, tot_weight;

      int32 num_sequences = NumSequencesInChainEg(io.indexes);
      KALDI_ASSERT(io.features.NumRows() % num_sequences == 0);
      int32 frames_per_sequence = io.features.NumRows() / num_sequences;
      ComputeKLObjfAndDeriv(opts_.chain_config, den_graph_,
                            io.features, 1.0, nnet_output,
                            num_sequences, frames_per_sequence,
                            &tot_objf, &tot_l2_term, &tot_weight,
                            &nnet_output_deriv,
                            (use_xent ? &xent_deriv : NULL));

      BaseFloat objf_scale = 1.0;
      {
        unordered_map<std::string, BaseFloat, StringHasher>::iterator it =
          objective_scales_.find(io.name);

        if (it != objective_scales_.end()) {
          objf_scale = it->second;
          tot_objf *= it->second;
          tot_l2_term *= it->second;
          tot_weight *= it->second;
          nnet_output_deriv.Scale(it->second);
        }
      }

      if (use_xent) {
        // this block computes the cross-entropy objective.
        const CuMatrixBase<BaseFloat> &xent_output = computer->GetOutput(
          xent_name);
        // at this point, xent_deriv is posteriors derived from the numerato
        // computation.  note, xent_objf has a factor of '.supervision.weight'
        BaseFloat xent_objf = TraceMatMat(xent_output, xent_deriv, kTrans);

        {
          unordered_map<std::string, BaseFloat, StringHasher>::iterator it =
            objective_scales_.find(xent_name);

          if (it != objective_scales_.end()) {
            xent_objf *= it->second;
            xent_deriv.Scale(it->second);
          }
        }

        objf_info_[xent_name + suffix].UpdateStats(xent_name + suffix,
                                          opts_.nnet_config.print_interval,
                                          num_minibatches_processed_,
                                          tot_weight, xent_objf);
      }

      if (opts_.apply_deriv_weights && io.deriv_weights.Dim() > 0) {
        CuVector<BaseFloat> cu_deriv_weights(io.deriv_weights);
        nnet_output_deriv.MulRowsVec(cu_deriv_weights);
        if (use_xent)
          xent_deriv.MulRowsVec(cu_deriv_weights);
      }

      std::vector<double> objective_values;
      objective_values.push_back(tot_l2_term);

      {
        unordered_map<std::string, ObjectiveFunctionInfo, StringHasher>::iterator it
          = objf_info_.find(io.name + suffix);

        if (it == objf_info_.end()) {
          std::vector<BaseFloat> aux_objf_scales(1, objf_scale);  // l2_term

          ObjectiveFunctionInfo totals(objf_scale, aux_objf_scales);
          it = objf_info_.insert(it, std::make_pair(io.name + suffix, totals));
        }

        if (opts_.accumulate_avg_deriv &&
            it->second.deriv_sum.Dim() == 0)
          it->second.deriv_sum.Resize(nnet_output.NumCols());

        if (it->second.deriv_sum.Dim() > 0)
          it->second.deriv_sum.AddRowSumMat(1.0, nnet_output_deriv, 1.0);

        it->second.UpdateStats(io.name + suffix,
                               opts_.nnet_config.print_interval,
                               num_minibatches_processed_,
                               tot_weight, tot_objf, objective_values);
      }

      computer->AcceptInput(io.name, &nnet_output_deriv);

      if (use_xent) {
        xent_deriv.Scale(opts_.chain_config.xent_regularize);
        if (opts_.accumulate_avg_deriv &&
            objf_info_[xent_name + suffix].deriv_sum.Dim() == 0)
          objf_info_[xent_name + suffix].deriv_sum.Resize(nnet_output.NumCols());
        if (objf_info_[xent_name + suffix].deriv_sum.Dim() > 0)
          objf_info_[xent_name + suffix].deriv_sum.AddRowSumMat(
              1.0, xent_deriv, 1.0);
        computer->AcceptInput(xent_name, &xent_deriv);
      }
    }
  }
}
*/

void NnetChainTrainer::ProcessOutputs(bool is_backstitch_step2,
                                      const NnetChainExample &eg,
                                      NnetComputer *computer) {
  // normally the eg will have just one output named 'output', but
  // we don't assume this.
  // In backstitch training, the output-name with the "_backstitch" suffix is
  // the one computed after the first, backward step of backstitch.
  const std::string suffix = (is_backstitch_step2 ? "_backstitch" : "");
  std::vector<NnetChainSupervision>::const_iterator iter = eg.outputs.begin(),
      end = eg.outputs.end();
  for (; iter != end; ++iter) {
    const NnetChainSupervision &sup = *iter;
    int32 node_index = nnet_->GetNodeIndex(sup.name);
    if (node_index < 0 ||
        !nnet_->IsOutputNode(node_index))
      KALDI_ERR << "Network has no output named " << sup.name;

    const CuMatrixBase<BaseFloat> &nnet_output = computer->GetOutput(sup.name);
    CuMatrix<BaseFloat> nnet_output_deriv(nnet_output.NumRows(),
                                          nnet_output.NumCols(),
                                          kUndefined);

    chain::ChainTrainingOptions chain_config(opts_.chain_config);

    {
      auto it = smbr_factors_.find(sup.name);
      if (it != smbr_factors_.end())
        chain_config.smbr_factor = it->second;
    }
    {
      auto it = mmi_factors_.find(sup.name);
      if (it != mmi_factors_.end())
        chain_config.mmi_factor = it->second;
    }
    {
      auto it = ml_factors_.find(sup.name);
      if (it != ml_factors_.end())
        chain_config.ml_factor = it->second;
    }

    bool use_xent = (chain_config.xent_regularize != 0.0);
    std::string xent_name = sup.name + "-xent";  // typically "output-xent".
    CuMatrix<BaseFloat> xent_deriv;

    BaseFloat tot_objf, tot_mmi_objf, tot_l2_term, tot_weight;

    if (sup.supervision.numerator_post_targets.NumRows() > 0) {
      ComputeKLObjfAndDeriv(opts_.chain_config, den_graph_,
                            sup.supervision, nnet_output,
                            &tot_objf, &tot_l2_term, &tot_weight,
                            &nnet_output_deriv,
                            (use_xent ? &xent_deriv : NULL));
    } else {
      if (chain_config.use_smbr_objective) {
        ComputeChainSmbrObjfAndDeriv(chain_config, den_graph_,
                                     sup.supervision, nnet_output,
                                     &tot_objf, &tot_mmi_objf, 
                                     &tot_l2_term, &tot_weight,
                                     &nnet_output_deriv,
                                     (use_xent ? &xent_deriv : NULL),
                                     sil_indices_.Dim() ? &sil_indices_ : NULL);
      } else {
        ComputeChainObjfAndDeriv(chain_config, den_graph_,
                                 sup.supervision, nnet_output,
                                 &tot_objf, &tot_l2_term, &tot_weight,
                                 &nnet_output_deriv,
                                 (use_xent ? &xent_deriv : NULL));
      }
    }
    if (use_xent) {
      // this block computes the cross-entropy objective.
      const CuMatrixBase<BaseFloat> &xent_output = computer->GetOutput(
          xent_name);
      // at this point, xent_deriv is posteriors derived from the numerator
      // computation.  note, xent_objf has a factor of '.supervision.weight'
      BaseFloat xent_objf = TraceMatMat(xent_output, xent_deriv, kTrans);

      objf_info_[xent_name + suffix].UpdateStats(xent_name + suffix,
                                        opts_.nnet_config.print_interval,
                                        num_minibatches_processed_,
                                        tot_weight, xent_objf);
    }

    if (opts_.apply_deriv_weights && sup.deriv_weights.Dim() != 0) {
      CuVector<BaseFloat> cu_deriv_weights(sup.deriv_weights);
      nnet_output_deriv.MulRowsVec(cu_deriv_weights);
      if (use_xent)
        xent_deriv.MulRowsVec(cu_deriv_weights);
    }

    std::vector<double> objective_values;
    objective_values.push_back(tot_l2_term);
    if (chain_config.use_smbr_objective)
      objective_values.push_back(tot_mmi_objf);

    {
      unordered_map<std::string, ObjectiveFunctionInfo, StringHasher>::iterator it
        = objf_info_.find(sup.name + suffix);

      if (it == objf_info_.end()) {
        BaseFloat this_objf_scale = 1.0;
        std::vector<BaseFloat> aux_objf_scales(1, 1.0);  // l2_term
        if (chain_config.use_smbr_objective) {
          this_objf_scale *= chain_config.smbr_factor;
          aux_objf_scales.push_back(
              (chain_config.mmi_factor + chain_config.ml_factor));
        }

        ObjectiveFunctionInfo totals(this_objf_scale, aux_objf_scales);
        it = objf_info_.insert(it, std::make_pair(sup.name + suffix, totals));
      }

      if (opts_.accumulate_avg_deriv &&
          it->second.deriv_sum.Dim() == 0)
        it->second.deriv_sum.Resize(nnet_output.NumCols());

      if (it->second.deriv_sum.Dim() > 0)
        it->second.deriv_sum.AddRowSumMat(1.0, nnet_output_deriv, 1.0);

      it->second.UpdateStats(sup.name + suffix,
                             opts_.nnet_config.print_interval,
                             num_minibatches_processed_,
                             tot_weight, tot_objf, objective_values);
    }

    computer->AcceptInput(sup.name, &nnet_output_deriv);

    if (use_xent) {
      xent_deriv.Scale(chain_config.xent_regularize);
      if (opts_.accumulate_avg_deriv &&
          objf_info_[xent_name + suffix].deriv_sum.Dim() == 0)
        objf_info_[xent_name + suffix].deriv_sum.Resize(nnet_output.NumCols());
      if (objf_info_[xent_name + suffix].deriv_sum.Dim() > 0)
        objf_info_[xent_name + suffix].deriv_sum.AddRowSumMat(
            1.0, xent_deriv, 1.0);
      computer->AcceptInput(xent_name, &xent_deriv);
    }
  }
}

bool NnetChainTrainer::PrintTotalStats() const {
  unordered_map<std::string, ObjectiveFunctionInfo, StringHasher>::const_iterator
      iter = objf_info_.begin(),
      end = objf_info_.end();
  bool ans = false;
  for (; iter != end; ++iter) {
    const std::string &name = iter->first;
    const ObjectiveFunctionInfo &info = iter->second;

    ans = info.PrintTotalStats(name) || ans;
  }
  PrintMaxChangeStats();
  return ans;
}

void NnetChainTrainer::PrintMaxChangeStats() const {
  KALDI_ASSERT(delta_nnet_ != NULL);
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  int32 i = 0;
  for (int32 c = 0; c < delta_nnet_->NumComponents(); c++) {
    Component *comp = delta_nnet_->GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
                  << "UpdatableComponent; change this code.";
      if (num_max_change_per_component_applied_[i] > 0)
        KALDI_LOG << "For " << delta_nnet_->GetComponentName(c)
                  << ", per-component max-change was enforced "
                  << (100.0 * num_max_change_per_component_applied_[i]) /
                     (num_minibatches_processed_ *
                     (nnet_config.backstitch_training_scale == 0.0 ? 1.0 :
                     1.0 + 1.0 / nnet_config.backstitch_training_interval))
                  << " \% of the time.";
      i++;
    }
  }
  if (num_max_change_global_applied_ > 0)
    KALDI_LOG << "The global max-change was enforced "
              << (100.0 * num_max_change_global_applied_) /
                 (num_minibatches_processed_ *
                 (nnet_config.backstitch_training_scale == 0.0 ? 1.0 :
                 1.0 + 1.0 / nnet_config.backstitch_training_interval))
              << " \% of the time.";
}

NnetChainTrainer::~NnetChainTrainer() {
  if (opts_.nnet_config.write_cache != "") {
    Output ko(opts_.nnet_config.write_cache, opts_.nnet_config.binary_write_cache);
    compiler_.WriteCache(ko.Stream(), opts_.nnet_config.binary_write_cache);
    KALDI_LOG << "Wrote computation cache to " << opts_.nnet_config.write_cache;
  }
  delete delta_nnet_;
}


} // namespace nnet3
} // namespace kaldi
