// nnet3/nnet-diagnostics.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/nnet-diagnostics.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetComputeProb::NnetComputeProb(const NnetComputeProbOptions &config,
                                 const Nnet &nnet):
    config_(config),
    nnet_(nnet),
    deriv_nnet_(NULL),
    compiler_(nnet),
    num_minibatches_processed_(0) {
  if (config_.compute_deriv) {
    deriv_nnet_ = new Nnet(nnet_);
    bool is_gradient = true;  // force simple update
    SetZero(is_gradient, deriv_nnet_);
  }
  if (!config.objective_scales_str.empty()) {

    std::vector<std::string> objective_scales;
    SplitStringToVector(config.objective_scales_str, ":", 
                        false, &objective_scales);
    if (objective_scales.size() %2 != 0) {
      KALDI_ERR << "Incorrect format for objective-scales-str " 
                << config.objective_scales_str;
    }
    
    for (int32 i = 0; i < objective_scales.size(); i += 2) {
      std::string &output_name = objective_scales[i];
      BaseFloat scale;

      if (!ConvertStringToReal(objective_scales[i+1], &scale)) {
        KALDI_ERR << "Could not convert objective-scale " 
                  << objective_scales[i+1] << " to float.";
      }

      objective_scales_[output_name] = scale;
    }
  }
}

const Nnet &NnetComputeProb::GetDeriv() const {
  if (deriv_nnet_ == NULL)
    KALDI_ERR << "GetDeriv() called when no derivatives were requested.";
  return *deriv_nnet_;
}

NnetComputeProb::~NnetComputeProb() {
  delete deriv_nnet_;  // delete does nothing if pointer is NULL.
}

void NnetComputeProb::Reset() {
  num_minibatches_processed_ = 0;
  objf_info_.clear();
  accuracy_info_.clear();
  if (deriv_nnet_) {
    bool is_gradient = true;
    SetZero(is_gradient, deriv_nnet_);
  }
}

void NnetComputeProb::Compute(const NnetExample &eg) {
  bool need_model_derivative = config_.compute_deriv,
      store_component_stats = false;
  ComputationRequest request;
  
  bool add_regularizer = config_.add_regularizer;

  GetComputationRequest(nnet_, eg, need_model_derivative,
                        store_component_stats, add_regularizer,
                        &request);
  const NnetComputation *computation = compiler_.Compile(request);
  NnetComputer computer(config_.compute_config, *computation,
                        nnet_, deriv_nnet_);
  // give the inputs to the computer object.
  computer.AcceptInputs(nnet_, eg.io);
  computer.Forward();
  this->ProcessOutputs(eg, &computer);
  if (config_.compute_deriv)
    computer.Backward();
}

void NnetComputeProb::ProcessOutputs(const NnetExample &eg,
                                     NnetComputer *computer) {
  std::vector<NnetIo>::const_iterator iter = eg.io.begin(),
      end = eg.io.end();
  for (; iter != end; ++iter) {
    const NnetIo &io = *iter;
    int32 node_index = nnet_.GetNodeIndex(io.name);
    if (node_index < 0)
      KALDI_ERR << "Network has no output named " << io.name;
    ObjectiveType obj_type = nnet_.GetNode(node_index).u.objective_type;
    BaseFloat scale = 1.0;
    if (objective_scales_.count(io.name) > 0)
      scale = objective_scales_[io.name];
    if (nnet_.IsOutputNode(node_index)) {
      const CuMatrixBase<BaseFloat> &output = computer->GetOutput(io.name);
      if (output.NumCols() != io.features.NumCols()) {
        KALDI_ERR << "Nnet versus example output dimension (num-classes) "
                  << "mismatch for '" << io.name << "': " << output.NumCols()
                  << " (nnet) vs. " << io.features.NumCols() << " (egs)\n";
      }
      {
        BaseFloat tot_weight, tot_objf;
        bool supply_deriv = config_.compute_deriv;

        CuMatrix<BaseFloat> nnet_output_deriv(output.NumRows(), output.NumCols());

        ComputeObjectiveFunction(io.features, obj_type, io.name,
                                 output,
                                 &tot_weight, &tot_objf,
                                 supply_deriv ? &nnet_output_deriv : NULL);

        if (scale != 1.0) {
          tot_objf *= scale;
          nnet_output_deriv.Scale(scale);
        }
        if (supply_deriv)
          computer->AcceptOutputDeriv(io.name, &nnet_output_deriv);

        SimpleObjectiveInfo &totals = objf_info_[io.name];
        totals.tot_weight += tot_weight;
        totals.tot_objective += tot_objf;
      }
      if (obj_type == kLinear && config_.compute_accuracy) {
        BaseFloat tot_weight, tot_accuracy;
        ComputeAccuracy(io.features, output,
                        &tot_weight, &tot_accuracy);
        SimpleObjectiveInfo &totals = accuracy_info_[io.name];
        totals.tot_weight += tot_weight;
        totals.tot_objective += tot_accuracy;
      }
      
      if (config_.add_regularizer) {
        std::string reg_name = io.name + "-reg";
        int32 reg_node_index = nnet_.GetNodeIndex(reg_name);

        if (reg_node_index >= 0) {
          KALDI_ASSERT(nnet_.IsOutputNode(reg_node_index));

          BaseFloat regularizer_scale = 1.0;

          if (objective_scales_.count(reg_name) > 0)
            regularizer_scale = objective_scales_[reg_name];

          BaseFloat tot_reg_weight, tot_reg_objf;

          const CuMatrixBase<BaseFloat> &reg_output = computer->GetOutput(reg_name);
          CuMatrix<BaseFloat> reg_output_deriv(reg_output.NumRows(),
              reg_output.NumCols(),
              kUndefined);
          bool supply_deriv = config_.compute_deriv;

          ComputeRegularizer(obj_type, reg_name, reg_output,
              &tot_reg_weight, &tot_reg_objf,
              supply_deriv ? &reg_output_deriv : NULL);

          tot_reg_objf *= scale;

          if (supply_deriv) {
            // TODO: Add deriv weights
            if (regularizer_scale != 1.0) 
              reg_output_deriv.Scale(regularizer_scale);

            computer->AcceptOutputDeriv(reg_name, &reg_output_deriv);
          }

          SimpleObjectiveInfo &totals = objf_info_[reg_name];
          totals.tot_weight += tot_reg_weight;
          totals.tot_objective += tot_reg_objf;
        }
      }
      num_minibatches_processed_++;
    }
  }
}

bool NnetComputeProb::PrintTotalStats() const {
  bool ans = false;
  unordered_map<std::string, SimpleObjectiveInfo, StringHasher>::const_iterator
      iter, end;
  { // First print regular objectives
    iter = objf_info_.begin();
    end = objf_info_.end();
    for (; iter != end; ++iter) {
      const std::string &name = iter->first;
      int32 node_index = nnet_.GetNodeIndex(name);
      KALDI_ASSERT(node_index >= 0);
      ObjectiveType obj_type = nnet_.GetNode(node_index).u.objective_type;
      const SimpleObjectiveInfo &info = iter->second;
      KALDI_LOG << "Overall "
                << (obj_type == kLinear ? "log-likelihood" : "objective")
                << " for '" << name << "' is "
                << (info.tot_objective / info.tot_weight) << " per frame"
                << ", over " << info.tot_weight << " frames.";
      if (info.tot_weight > 0)
        ans = true;
    }
  }
  { // now print accuracies.
    iter = accuracy_info_.begin();
    end = accuracy_info_.end();
    for (; iter != end; ++iter) {
      const std::string &name = iter->first;
      const SimpleObjectiveInfo &info = iter->second;
      KALDI_LOG << "Overall accuracy for '" << name << "' is "
                << (info.tot_objective / info.tot_weight) << " per frame"
                << ", over " << info.tot_weight << " frames.";
      // don't bother changing ans; the loop over the regular objective should
      // already have set it to true if we got any data.
    }
  }
  return ans;
}

void ComputeAccuracy(const GeneralMatrix &supervision,
                     const CuMatrixBase<BaseFloat> &nnet_output,
                     BaseFloat *tot_weight_out,
                     BaseFloat *tot_accuracy_out) {
  int32 num_rows = nnet_output.NumRows(),
      num_cols = nnet_output.NumCols();
  KALDI_ASSERT(supervision.NumRows() == num_rows &&
               supervision.NumCols() == num_cols);

  CuArray<int32> best_index(num_rows);
  nnet_output.FindRowMaxId(&best_index);
  std::vector<int32> best_index_cpu;
  // wasteful copy, but doesn't dominate.
  best_index.CopyToVec(&best_index_cpu);


  double tot_weight = 0.0,
      tot_accuracy = 0.0;

  // note: we expect that in most cases where this code is called,
  // supervision.Type() will be kSparseMatrix.
  switch (supervision.Type()) {
    case kCompressedMatrix: {
      Matrix<BaseFloat> mat;
      supervision.GetMatrix(&mat);
      for (int32 r = 0; r < num_rows; r++) {
        SubVector<BaseFloat> vec(mat, r);
        BaseFloat row_sum = vec.Sum();
        KALDI_ASSERT(row_sum >= 0.0);
        int32 best_index;
        vec.Max(&best_index);  // discard max value.
        tot_weight += row_sum;
        if (best_index == best_index_cpu[r])
          tot_accuracy += row_sum;
      }
      break;

    }
    case kFullMatrix: {
      const Matrix<BaseFloat> &mat = supervision.GetFullMatrix();
      for (int32 r = 0; r < num_rows; r++) {
        SubVector<BaseFloat> vec(mat, r);
        BaseFloat row_sum = vec.Sum();
        KALDI_ASSERT(row_sum >= 0.0);
        int32 best_index;
        vec.Max(&best_index);  // discard max value.
        tot_weight += row_sum;
        if (best_index == best_index_cpu[r])
          tot_accuracy += row_sum;
      }
      break;
    }
    case kSparseMatrix: {
      const SparseMatrix<BaseFloat> &smat = supervision.GetSparseMatrix();
      for (int32 r = 0; r < num_rows; r++) {
        const SparseVector<BaseFloat> &row = smat.Row(r);
        BaseFloat row_sum = row.Sum();
        int32 best_index;
        row.Max(&best_index);
        KALDI_ASSERT(best_index < num_cols);
        tot_weight += row_sum;
        if (best_index == best_index_cpu[r])
          tot_accuracy += row_sum;
      }
      break;
    }
    default: KALDI_ERR << "Bad general-matrix type.";
  }
  *tot_weight_out = tot_weight;
  *tot_accuracy_out = tot_accuracy;
}

const SimpleObjectiveInfo* NnetComputeProb::GetObjective(
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
