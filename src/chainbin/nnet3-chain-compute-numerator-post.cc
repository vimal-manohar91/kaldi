// nnet3bin/nnet3-chain-compute-numerator-post.cc

// Copyright 2018  Vimal Manohar

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-chain-diagnostics.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

void ProcessOutputs(const Nnet &nnet,
                    const NnetChainExample &eg, NnetComputer *computer,
                    NnetChainExample *eg_out) {
  *eg_out = eg;

  // There will normally be just one output here, named 'output',
  // but the code is more general than this.
  std::vector<NnetChainSupervision>::const_iterator iter = eg.outputs.begin(),
      end = eg.outputs.end();
  std::vector<NnetChainSupervision>::iterator out_iter = eg_out->outputs.begin(),
      out_end = eg_out->outputs.end();
  for (; iter != end; ++iter, ++out_iter) {
    const NnetChainSupervision &sup = *iter;
    int32 node_index = nnet.GetNodeIndex(sup.name);
    if (node_index < 0 ||
        !nnet.IsOutputNode(node_index))
      KALDI_ERR << "Network has no output named " << sup.name;

    const CuMatrixBase<BaseFloat> &nnet_output = computer->GetOutput(sup.name);

    CuMatrix<BaseFloat> numerator_post(
        nnet_output.NumRows(), nnet_output.NumCols(), kUndefined);
    chain::ComputeChainNumeratorPost(sup.supervision,
                                     nnet_output, &numerator_post);

    out_iter->supervision.numerator_post_targets =
      SparseMatrix<BaseFloat>(Matrix<BaseFloat>(numerator_post));
  }
}

void ComputeNumeratorPost(const NnetComputeProbOptions &nnet_config,
                          const Nnet &nnet,
                          CachingOptimizingCompiler *compiler,
                          const NnetChainExample &eg,
                          NnetChainExample *eg_out) {
  bool need_model_derivative = false, store_component_stats = false,
      use_xent_regularization = false, use_xent_derivative = false;

  ComputationRequest request;
  GetChainComputationRequest(nnet, eg, need_model_derivative,
                             store_component_stats, use_xent_regularization,
                             use_xent_derivative, &request);

  std::shared_ptr<const NnetComputation> computation = compiler->Compile(request);
  NnetComputer computer(nnet_config.compute_config, *computation,
                        nnet, NULL);
  // give the inputs to the computer object.
  computer.AcceptInputs(nnet, eg.inputs);
  computer.Run();
  ProcessOutputs(nnet, eg, &computer, eg_out);
}

}   // namespace nnet3
}   // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Computes the numerator posteriors per frame of the given data with \n"
        "an nnet3+chain neural net and outputs egs that include those \n"
        "numerator posteriors.  The input of this is the output of\n"
        "e.g. nnet3-chain-get-egs |\n"
        "\n"
        "Usage:  nnet3-chain-compute-numerator-post [options] <raw-nnet3-model-in> <training-examples-in> <training-examples-out>\n"
        "e.g.: nnet3-chain-compute-numerator-post 0.mdl ark:cegs.1.ark ark:cegs_out.1.ark\n";

    bool batchnorm_test_mode = true, dropout_test_mode = true;

    // This program doesn't support using a GPU, because these probabilities are
    // used for diagnostics, and you can just compute them with a small enough
    // amount of data that a CPU can do it within reasonable time.
    // It wouldn't be hard to make it support GPU, though.

    NnetComputeProbOptions nnet_opts;

    ParseOptions po(usage);

    po.Register("batchnorm-test-mode", &batchnorm_test_mode,
                "If true, set test-mode to true on any BatchNormComponents.");
    po.Register("dropout-test-mode", &dropout_test_mode,
                "If true, set test-mode to true on any DropoutComponents and "
                "DropoutMaskComponents.");

    nnet_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    nnet_opts.compute_deriv = false;

    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        examples_wspecifier = po.GetArg(3);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    if (batchnorm_test_mode)
      SetBatchnormTestMode(true, &nnet);

    if (dropout_test_mode)
      SetDropoutTestMode(true, &nnet);

    SequentialNnetChainExampleReader example_reader(examples_rspecifier);
    NnetChainExampleWriter example_writer(examples_wspecifier);

    CachingOptimizingCompiler compiler(nnet, nnet_opts.optimize_config,
                                       nnet_opts.compiler_config);

    int32 num_done = 0;
    for (; !example_reader.Done(); example_reader.Next()) {
      NnetChainExample eg_out;
      ComputeNumeratorPost(nnet_opts, nnet, &compiler,
                           example_reader.Value(), &eg_out);
      example_writer.Write(example_reader.Key(), eg_out);
      num_done++;
    }

    return (num_done > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

