// nnet3bin/nnet3-chain-train.cc

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
#include "nnet3/nnet-chain-training.h"
#include "chain/chain-den-graph.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    using namespace kaldi::chain;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train nnet3+chain neural network parameters with backprop and stochastic\n"
        "gradient descent.  Minibatches are to be created by nnet3-chain-merge-egs in\n"
        "the input pipeline.  This training program is single-threaded (best to\n"
        "use it with a GPU).\n"
        "\n"
        "Usage:  nnet3-chain-train [options] <raw-nnet-in> [<denominator-fst1-in> ...] <chain-training-examples-in> <raw-nnet-out>\n"
        "\n"
        "nnet3-chain-train 1.raw den.fst 'ark:nnet3-merge-egs 1.cegs ark:-|' 2.raw\n";

    bool binary_write = true;
    std::string use_gpu = "yes";
    NnetChainTrainingOptions opts;
    std::string den_fst_to_output_str;

    ParseOptions po(usage);
    po.Register("den-fst-to-output", &den_fst_to_output_str, "Comma-separated string of output-names "
                "correspond to list of den_fsts. If not specified den_fsts assigend "
                "to outputs with name output-0,.. respectively.");
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    int32 num_args = po.NumArgs(),
      num_den_fst = num_args - 3;
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif
    std::vector<std::string> den_fst_rxfilenames(num_den_fst);
    for (int32 fst_ind = 0; fst_ind < num_den_fst; fst_ind++)
      den_fst_rxfilenames[fst_ind] = po.GetArg(fst_ind+2);

    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(num_args-1),
        nnet_wxfilename = po.GetArg(num_args);


    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    bool ok;

    {
      std::vector<fst::StdVectorFst> den_fst(num_den_fst);
      std::vector<std::string> den_fst_to_output;
      if (!den_fst_to_output_str.empty()) {
        SplitStringToVector(den_fst_to_output_str, ",", true, &den_fst_to_output);
        KALDI_ASSERT(den_fst_to_output.size() == num_den_fst);
      } else {
        if (num_den_fst == 1) {
          den_fst_to_output.push_back("output");
        } else {
          for (int32 fst_ind = 0; fst_ind < num_den_fst; fst_ind++)
            den_fst_to_output.push_back("output"+std::to_string(fst_ind));
        }
      }

      for (int32 fst_ind = 0; fst_ind < num_den_fst; fst_ind++)
        ReadFstKaldi(den_fst_rxfilenames[fst_ind], &den_fst[fst_ind]);

      NnetChainTrainer trainer(opts, den_fst, den_fst_to_output, &nnet);

      SequentialNnetChainExampleReader example_reader(examples_rspecifier);

      for (; !example_reader.Done(); example_reader.Next())
        trainer.Train(example_reader.Value());

      ok = trainer.PrintTotalStats();
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    WriteKaldiObject(nnet, nnet_wxfilename, binary_write);
    KALDI_LOG << "Wrote raw model to " << nnet_wxfilename;
    return (ok ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
