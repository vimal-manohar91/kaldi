// ivectorbin/ivector-sum-plda.cc

// Copyright 2013  Daniel Povey
//           2017  Vimal Manohar

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
#include "ivector/plda.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Sum two PLDA models\n"
        "Usage:  ivector-sum-plda [options] <plda1> <plda2> <plda-out>\n"
        "e.g.: \n"
        " ivector-sum-plda plda1 plda2 plda\n";
    
    BaseFloat scale1 = 1.0, scale2 = 1.0;
    bool binary = true;
    ParseOptions po(usage);
    
    po.Register("scale1", &scale1, "Scale applied to first plda");
    po.Register("scale2", &scale2, "Scale applied to second plda");
    po.Register("binary", &binary, "If true, write output as binary.");

    PldaConfig plda_config;
    plda_config.Register(&po);

    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda_rxfilename1 = po.GetArg(1),
      plda_rxfilename2 = po.GetArg(2),
      plda_wxfilename = po.GetArg(3);

    Plda plda1;
    ReadKaldiObject(plda_rxfilename1, &plda1);
    Plda plda2;
    ReadKaldiObject(plda_rxfilename2, &plda2);

    plda1.Sum(scale2, plda2, scale1);
    WriteKaldiObject(plda1, plda_wxfilename, binary);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


