// nnet2bin/nnet-train-discriminative-unsupervised-parallel.cc

// Copyright 2015  Vimal Manohar (Johns Hopkins University)

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
#include "hmm/transition-model.h"
#include "nnet2/nnet-example-functions.h"
#include "nnet2/am-nnet.h"
#include "nnet2/nnet-compute-discriminative-unsupervised-parallel.h"
#include "base/kaldi-types-extra.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    typedef kaldi::SignedLogReal<double> SignedLogDouble;

    const char *usage =
        "Train the neural network parameters with a discriminative objective\n"
        "function (NCE) in an unsupervised way.  This uses training examples\n"
        "prepared with nnet-get-egs-discriminative-unsupervised\n"
        "This version used multiple threads (but no GPU)"
        "\n"
        "Usage:  nnet-train-discriminative-unsupervised-parallel [options] <model-in> <training-examples-in> <model-out>\n"
        "e.g.:\n"
        "nnet-train-discriminative-unsupervised-parallel 1.nnet ark:1.uegs 2.nnet\n";
    
    bool binary_write = true;
    std::string use_gpu = "yes";
    int32 num_threads = 1;
    int32 pdf_id = -1;
    bool store_gradients = false;
    NnetDiscriminativeUnsupervisedUpdateOptions update_opts;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("num-threads", &num_threads, "Number of threads to use");
    po.Register("print-gradient-for-pdf", &pdf_id, "For debugging");
    po.Register("store-gradients", &store_gradients, "Store gradients for debugging");
    
    update_opts.Register(&po);

    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);

    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }

    NnetDiscriminativeUnsupervisedStats stats(trans_model.NumPdfs());
    stats.store_gradients = store_gradients;
    stats.logit_stats = store_gradients;

    if (pdf_id >= 0) {
      stats.store_gradients = true;
      stats.logit_stats = true;
    }

    SequentialDiscriminativeUnsupervisedNnetExampleReader example_reader(examples_rspecifier);

    NnetDiscriminativeUnsupervisedUpdateParallel(am_nnet, trans_model,
        update_opts, num_threads, &example_reader,
        &(am_nnet.GetNnet()), &stats);

    stats.Print(update_opts.criterion, true, true);
    
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }

    return (stats.tot_t == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


