// nnet3bin/nnet3-chain-compute-prob.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Computes and prints to in logging messages the average log-prob per frame of\n"
        "the given data with an nnet3+chain neural net.  The input of this is the output of\n"
        "e.g. nnet3-chain-get-egs | nnet3-chain-merge-egs.\n"
        "\n"
        "Usage:  nnet3-chain-compute-prob [options] <raw-nnet3-model-in> <denominator-fst-1> [<denominator-fst-2> ...] <training-examples-in>\n"
        "e.g.: nnet3-chain-compute-prob 0.mdl den.fst ark:valid.egs\n";

    bool batchnorm_test_mode = true, dropout_test_mode = true;
    std::string den_fst_to_outputs_str;

    // This program doesn't support using a GPU, because these probabilities are
    // used for diagnostics, and you can just compute them with a small enough
    // amount of data that a CPU can do it within reasonable time.
    // It wouldn't be hard to make it support GPU, though.

    NnetComputeProbOptions nnet_opts;
    chain::ChainTrainingOptions chain_opts;

    ParseOptions po(usage);

    po.Register("batchnorm-test-mode", &batchnorm_test_mode,
                "If true, set test-mode to true on any BatchNormComponents.");
    po.Register("dropout-test-mode", &dropout_test_mode,
                "If true, set test-mode to true on any DropoutComponents and "
                "DropoutMaskComponents.");
    po.Register("den-fst-to-outputs", &den_fst_to_outputs_str, 
                "A space-separated list of comma-separated list of output-names "
                "corresponding to the list of den_fsts. If not specified, "
                "then only one denominator fst is expected corresponding to "
                "output 'output'");

    nnet_opts.Register(&po);
    chain_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::vector<std::vector<std::string> > den_fst_to_outputs;
    int32 num_den_fsts = 1;
    if (!den_fst_to_outputs_str.empty()) {
      std::vector<std::string> den_fst_to_output_list;
      SplitStringToVector(den_fst_to_outputs_str, " ", true, &den_fst_to_output_list);
      num_den_fsts = den_fst_to_output_list.size();
      den_fst_to_outputs.resize(num_den_fsts);
      for (int32 i = 0; i < num_den_fsts; i++) {
        const std::string &output_list = den_fst_to_output_list[i];
        SplitStringToVector(output_list, ",;", true, &den_fst_to_outputs[i]);
      }
    } else {
      den_fst_to_outputs.push_back(std::vector<std::string>(1, "output"));
    }
    KALDI_ASSERT(num_den_fsts == po.NumArgs() - 2);

    std::vector<std::string> den_fst_rxfilenames(num_den_fsts);
    for (int32 fst_ind = 0; fst_ind < num_den_fsts; fst_ind++)
      den_fst_rxfilenames[fst_ind] = po.GetArg(fst_ind + 2);

    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(po.NumArgs());

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    if (batchnorm_test_mode)
      SetBatchnormTestMode(true, &nnet);

    if (dropout_test_mode)
      SetDropoutTestMode(true, &nnet);

    std::vector<fst::StdVectorFst> den_fsts(num_den_fsts);
    for (int32 fst_ind = 0; fst_ind < num_den_fsts; fst_ind++)
      ReadFstKaldi(den_fst_rxfilenames[fst_ind], &den_fsts[fst_ind]);

    if (GetVerboseLevel() > 2)
      nnet_opts.compute_deriv = true;

    NnetChainComputeProb chain_prob_computer(nnet_opts, chain_opts, 
                                             den_fsts, den_fst_to_outputs,
                                             nnet);

    SequentialNnetChainExampleReader example_reader(examples_rspecifier);

    for (; !example_reader.Done(); example_reader.Next())
      chain_prob_computer.Compute(example_reader.Value());

    bool ok = chain_prob_computer.PrintTotalStats();

    return (ok ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
