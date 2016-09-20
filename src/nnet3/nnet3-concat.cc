// nnet3bin/nnet3-concat.cc

// Copyright 2016 Vimal Manohar

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
#include "nnet3/nnet-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Concat multiple 'raw' nnets and write a concatenated 'raw' nnet\n"
        "Similar to repeatedly applying nnet3-init, but this "
        "copies the network components in addition to the configs\n"
        "\n"
        "Usage:  nnet3-concat [options] <output-raw-nnet> <raw-nnet1> <raw-nnet2> [... <raw-nnetN>]\n"
        "e.g.:\n"
        " nnet3-concat concat.raw foo.raw bar.raw\n"
        "See also: nnet3-init, nnet3-copy, nnet3-info\n";
    
    bool binary_write = true;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string raw_nnet_wxfilename = po.GetArg(1);
    std::string raw_nnet_rxfilename = po.GetArg(2);
  
    int32 num_nnets = po.NumArgs() - 1;
    
    Nnet nnet;
    {
      ReadKaldiObject(raw_nnet_rxfilename, &nnet);
      KALDI_LOG << "Read raw neural net from "
                << raw_nnet_rxfilename;
    }
    
    for (int32 n = 3; n <= po.NumArgs(); n++) {
      bool binary;
      Input ki(po.GetArg(n), &binary);
      nnet.ConcatNnet(ki.Stream(), binary);
    }

    WriteKaldiObject(nnet, raw_nnet_wxfilename, binary_write);
    KALDI_LOG << "Initialized raw neural net and wrote it to "
              << raw_nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

