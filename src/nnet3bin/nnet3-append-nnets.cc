// nnet3bin/nnet3-append-nnets.cc

// Copyright 2016  Vimal Manohar

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Concatenates two nnet3 neural networks; outputs 'raw' nnet\n"
        "Similar to nnet3-init, but concatenates the parameters in addition "
        "to the configs\n"
        "\n"
        "Usage:  nnet3-append-nnets [options] <nnet-out> <nnet-in1> <nnet-in2> [<nnet-inN>\n"
        "e.g.:\n"
        " nnet3-append-nnets --binary=false out.raw in1.raw in2.raw\n";

    bool binary_write = true;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    po.Read(argc, argv);
    
    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string out_raw_nnet_wxfilename = po.GetArg(1);
    
    Nnet nnet;
    ReadKaldiObject(po.GetArg(2), &nnet);
    
    for (int32 i = 3; i <= po.NumArgs(); i++) {
      bool binary;
      Input ki(po.GetArg(i), &binary);
      nnet.ConcatNnet(ki.Stream(), binary);
    }

    WriteKaldiObject(nnet, out_raw_nnet_wxfilename, binary_write);
    KALDI_LOG << "Concatenated raw neural networks to "
              << out_raw_nnet_wxfilename;

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

