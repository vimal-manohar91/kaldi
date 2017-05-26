// chainbin/nnet3-chain-combine.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet3/nnet-chain-combine.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Using a subset of training or held-out nnet3+chain examples, compute an\n"
        "optimal combination of  anumber of nnet3 neural nets by maximizing the\n"
        "'chain' objective function.  See documentation of options for more details.\n"
        "Inputs and outputs are nnet3 raw nnets.\n"
        "\n"
        "Usage:  nnet3-chain-combine [options] [<den-fst-in1> ...] <raw-nnet-in1> <raw-nnet-in2> ... <raw-nnet-inN> <chain-examples-in> <raw-nnet-out>\n"
        "\n"
        "e.g.:\n"
        " nnet3-combine den.fst 35.raw 36.raw 37.raw 38.raw ark:valid.cegs final.raw\n";

    bool binary_write = true;
    std::string use_gpu = "yes";
    std::string den_fst_to_output_str;
    NnetCombineConfig combine_config;
    chain::ChainTrainingOptions chain_config;

    ParseOptions po(usage);
    po.Register("den-fst-to-output", &den_fst_to_output_str, "Comma-separated string of output-names "
                "correspond to list of den_fsts. If not specified den_fsts assigend "
                "to outputs with name output-0,.. respectively.");
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    combine_config.Register(&po);
    chain_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() < 4) {
      po.PrintUsage();
      exit(1);
    }
    std::vector<std::string> den_fst_to_output;
    int32 num_den_fst = 1;
    if (!den_fst_to_output_str.empty()) {
      SplitStringToVector(den_fst_to_output_str, ",", true, &den_fst_to_output);
    } else {
      den_fst_to_output.push_back("output");
    }

    num_den_fst = den_fst_to_output.size();
    std::vector<fst::StdVectorFst> den_fst(num_den_fst);
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif
    std::vector<std::string> den_fst_rxfilenames(num_den_fst);
    for (int32 fst_ind = 0; fst_ind < num_den_fst; fst_ind++)
      den_fst_rxfilenames[fst_ind] = po.GetArg(fst_ind+1);
    std::string
        raw_nnet_rxfilename = po.GetArg(po.NumArgs() - 2),
        valid_examples_rspecifier = po.GetArg(po.NumArgs() - 1),
        nnet_wxfilename = po.GetArg(po.NumArgs());


    for (int32 fst_ind = 0; fst_ind < num_den_fst; fst_ind++)
      ReadFstKaldi(den_fst_rxfilenames[fst_ind], &den_fst[fst_ind]);

    Nnet nnet;
    ReadKaldiObject(raw_nnet_rxfilename, &nnet);


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


    int32 num_nnets = po.NumArgs() - 2 - num_den_fst;
    NnetChainCombiner combiner(combine_config, chain_config,
                               num_nnets, egs, den_fst, den_fst_to_output, nnet);

    for (int32 n = 1; n < num_nnets; n++) {
      std::string this_nnet_rxfilename = po.GetArg(n + 1 + num_den_fst);
      ReadKaldiObject(this_nnet_rxfilename, &nnet);
      combiner.AcceptNnet(nnet);
    }

    combiner.Combine();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    WriteKaldiObject(combiner.GetNnet(), nnet_wxfilename, binary_write);

    KALDI_LOG << "Finished combining neural nets, wrote model to "
              << nnet_wxfilename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


