// chainbin/nnet3-chain-combine.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//                2017  Yiming Wang

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
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-chain-diagnostics.h"


namespace kaldi {
namespace nnet3 {

// Computes and returns the objective function for the examples in 'egs' given
// the model in 'nnet'. If either of batchnorm/dropout test modes is true, we
// make a copy of 'nnet', set test modes on that and evaluate its objective.
// Note: the object that prob_computer->nnet_ refers to should be 'nnet'.
double ComputeObjf(bool batchnorm_test_mode, bool dropout_test_mode,
                   const std::vector<NnetChainExample> &egs, const Nnet &nnet,
                   const chain::ChainTrainingOptions &chain_config,
                   const std::vector<fst::StdVectorFst> &den_fsts,
                   const std::vector<std::vector<std::string> > &den_fst_to_outputs,
                   NnetChainComputeProb *prob_computer) {
  if (batchnorm_test_mode || dropout_test_mode) {
    Nnet nnet_copy(nnet);
    if (batchnorm_test_mode)
      SetBatchnormTestMode(true, &nnet_copy);
    if (dropout_test_mode)
      SetDropoutTestMode(true, &nnet_copy);
    NnetComputeProbOptions compute_prob_opts;
    NnetChainComputeProb prob_computer_test(compute_prob_opts, chain_config,
        den_fsts, den_fst_to_outputs, nnet_copy);
    return ComputeObjf(false, false, egs, nnet_copy,
                       chain_config, den_fsts, den_fst_to_outputs, &prob_computer_test);
  } else {
    prob_computer->Reset();
    std::vector<NnetChainExample>::const_iterator iter = egs.begin(),
                                                   end = egs.end();
    for (; iter != end; ++iter)
      prob_computer->Compute(*iter);

    double tot_weight = 0.0;
    double tot_objf = prob_computer->GetTotalObjective(&tot_weight);

    KALDI_ASSERT(tot_weight > 0.0);
    // inf/nan tot_objf->return -inf objective.
    if (!(tot_objf == tot_objf && tot_objf - tot_objf == 0))
      return -std::numeric_limits<double>::infinity();
    // we prefer to deal with normalized objective functions.
    return tot_objf / tot_weight;
  }
}

// Updates moving average over num_models nnets, given the average over
// previous (num_models - 1) nnets, and the new nnet.
void UpdateNnetMovingAverage(int32 num_models,
    const Nnet &nnet, Nnet *moving_average_nnet) {
  KALDI_ASSERT(NumParameters(nnet) == NumParameters(*moving_average_nnet));
  ScaleNnet((num_models - 1.0) / num_models, moving_average_nnet);
  AddNnet(nnet, 1.0 / num_models, moving_average_nnet);
}

}
}


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Using a subset of training or held-out nnet3+chain examples, compute\n"
        "the average over the first n nnet models where we maximize the\n"
        "'chain' objective function for n. Note that the order of models has\n"
        "been reversed before feeding into this binary. So we are actually\n"
        "combining last n models.\n"
        "Inputs and outputs are nnet3 raw nnets.\n"
        "\n"
        "Usage:  nnet3-chain-combine [options] <den-fst-1> [<den-fst-2> ...] <raw-nnet-in1> <raw-nnet-in2> ... <raw-nnet-inN> <chain-examples-in> <raw-nnet-out>\n"
        "\n"
        "e.g.:\n"
        " nnet3-combine den.fst 35.raw 36.raw 37.raw 38.raw ark:valid.cegs final.raw\n";

    bool binary_write = true;
    int32 max_objective_evaluations = 30;
    bool batchnorm_test_mode = false,
        dropout_test_mode = true;
    std::string use_gpu = "yes";
    std::string den_fst_to_outputs_str;
    chain::ChainTrainingOptions chain_config;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("max-objective-evaluations", &max_objective_evaluations, "The "
                "maximum number of objective evaluations in order to figure "
                "out the best number of models to combine. It helps to speedup "
                "if the number of models provided to this binary is quite "
                "large (e.g. several hundred)."); 
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("batchnorm-test-mode", &batchnorm_test_mode,
                "If true, set test-mode to true on any BatchNormComponents "
                "while evaluating objectives.");
    po.Register("dropout-test-mode", &dropout_test_mode,
                "If true, set test-mode to true on any DropoutComponents and "
                "DropoutMaskComponents while evaluating objectives.");
    po.Register("den-fst-to-outputs", &den_fst_to_outputs_str, 
                "A space-separated list of comma-separated list of output-names "
                "corresponding to the list of den_fsts. If not specified, "
                "then only one denominator fst is expected corresponding to "
                "output 'output'");

    chain_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() < 4) {
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

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif
    std::vector<std::string> den_fst_rxfilenames(num_den_fsts);
    for (int32 fst_ind = 0; fst_ind < num_den_fsts; fst_ind++)
      den_fst_rxfilenames[fst_ind] = po.GetArg(fst_ind + 1);
    std::string
        raw_nnet_rxfilename = po.GetArg(num_den_fsts + 1),
        valid_examples_rspecifier = po.GetArg(po.NumArgs() - 1),
        nnet_wxfilename = po.GetArg(po.NumArgs());

    std::vector<fst::StdVectorFst> den_fsts(num_den_fsts);
    for (int32 fst_ind = 0; fst_ind < num_den_fsts; fst_ind++)
      ReadFstKaldi(den_fst_rxfilenames[fst_ind], &den_fsts[fst_ind]);

    Nnet nnet;
    ReadKaldiObject(raw_nnet_rxfilename, &nnet);
    Nnet moving_average_nnet(nnet), best_nnet(nnet);
    NnetComputeProbOptions compute_prob_opts;
    NnetChainComputeProb prob_computer(compute_prob_opts, chain_config,
        den_fsts, den_fst_to_outputs, moving_average_nnet);

    std::vector<NnetChainExample> egs;
    egs.reserve(10000);  // reserve a lot of space to minimize the chance of
                         // reallocation.

    { // This block adds training examples to "egs".
      SequentialNnetChainExampleReader example_reader(
          valid_examples_rspecifier);
      for (; !example_reader.Done(); example_reader.Next())
        egs.push_back(example_reader.Value());
      KALDI_LOG << "Read " << egs.size() << " examples.";
      KALDI_ASSERT(!egs.empty());
    }

    // first evaluates the objective using the last model.
    int32 best_num_to_combine = 1;
    double
        init_objf = ComputeObjf(batchnorm_test_mode, dropout_test_mode,
            egs, moving_average_nnet, chain_config, 
            den_fsts, den_fst_to_outputs, &prob_computer),
        best_objf = init_objf;
    KALDI_LOG << "objective function using the last model is " << init_objf;

    int32 num_nnets = po.NumArgs() - num_den_fsts - 2;
    // then each time before we re-evaluate the objective function, we will add
    // num_to_add models to the moving average.
    int32 num_to_add = (num_nnets + max_objective_evaluations - 1) /
                       max_objective_evaluations;
    for (int32 n = 1; n < num_nnets; n++) {
      std::string this_nnet_rxfilename = po.GetArg(n + num_den_fsts + 1);
      ReadKaldiObject(this_nnet_rxfilename, &nnet);
      // updates the moving average
      UpdateNnetMovingAverage(n + 1, nnet, &moving_average_nnet);
      // evaluates the objective everytime after adding num_to_add model or
      // all the models to the moving average.
      if ((n - 1) % num_to_add == num_to_add - 1 || n == num_nnets - 1) {
        double objf = ComputeObjf(batchnorm_test_mode, dropout_test_mode,
            egs, moving_average_nnet, chain_config, 
            den_fsts, den_fst_to_outputs, &prob_computer);
        KALDI_LOG << "Combining last " << n + 1
                  << " models, objective function is " << objf;
        if (objf > best_objf) {
          best_objf = objf;
          best_nnet = moving_average_nnet;
          best_num_to_combine = n + 1;
        }
      }
    }
    KALDI_LOG << "Combining " << best_num_to_combine
              << " nnets, objective function changed from " << init_objf
              << " to " << best_objf;

    if (HasBatchnorm(nnet))
      RecomputeStats(egs, chain_config, den_fsts, den_fst_to_outputs, &best_nnet);

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    WriteKaldiObject(best_nnet, nnet_wxfilename, binary_write);
    KALDI_LOG << "Finished combining neural nets, wrote model to "
              << nnet_wxfilename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
