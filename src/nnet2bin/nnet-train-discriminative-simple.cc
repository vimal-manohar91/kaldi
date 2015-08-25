// nnet2bin/nnet-train-discriminative-simple.cc

// Copyright 2013       Johns Hopkins University (author: Daniel Povey)
// Copyright 2014-2015  Vimal Manohar

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
#include "nnet2/am-nnet.h"
#include "nnet2/nnet-compute-discriminative.h"
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
        "function (MMI, SMBR or MPFE).  This uses training examples prepared with\n"
        "nnet-get-egs-discriminative\n"
        "\n"
        "Usage:  nnet-train-discriminative-simple [options] <model-in> <training-examples-in> <model-out>\n"
        "e.g.:\n"
        "nnet-train-discriminative-simple 1.nnet ark:1.degs 2.nnet\n";
    
    bool binary_write = true;
    std::string use_gpu = "yes";
    bool store_gradients = false;
    int32 pdf_id = -1;
    
    NnetDiscriminativeUpdateOptions update_opts;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("print-gradient-for-pdf", &pdf_id, "For debugging");
    po.Register("store-gradients", &store_gradients, "Store gradients for debugging");

    update_opts.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);

    int64 num_examples = 0;

    TransitionModel trans_model;
    {
      AmNnet am_nnet;
      {
        bool binary_read;
        Input ki(nnet_rxfilename, &binary_read);
        trans_model.Read(ki.Stream(), binary_read);
        am_nnet.Read(ki.Stream(), binary_read);
      }

      NnetDiscriminativeStats stats(trans_model.NumPdfs());;
      stats.store_gradients = store_gradients;
      stats.store_logit_stats = store_gradients;

      if (pdf_id >= 0) {
        stats.store_gradients = true;
        stats.store_logit_stats = true;
      }

      SequentialDiscriminativeNnetExampleReader example_reader(examples_rspecifier);

      for (; !example_reader.Done(); example_reader.Next(), num_examples++) {
        NnetDiscriminativeUpdate(am_nnet, trans_model, update_opts,
                                 example_reader.Value(),
                                 &(am_nnet.GetNnet()), &stats);

        if (GetVerboseLevel() >= 5) 
          stats.Print(update_opts.criterion);
        else {
          if (num_examples % 10 == 0 && num_examples != 0) { // each example might be 500 frames.
            if (GetVerboseLevel() >= 2) {
              stats.Print(update_opts.criterion);
              if (pdf_id >= 0) {
                stats.PrintPost(pdf_id);
              }
            }
          }
        }
      }

      stats.Print(update_opts.criterion, true, true);
        
      {
        Output ko(nnet_wxfilename, binary_write);
        trans_model.Write(ko.Stream(), binary_write);
        am_nnet.Write(ko.Stream(), binary_write);
      }
    }
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    KALDI_LOG << "Finished training, processed " << num_examples
              << " training examples.  Wrote model to "
              << nnet_wxfilename;
    return (num_examples == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


