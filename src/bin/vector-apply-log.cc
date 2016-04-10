// bin/vector-apply-log.cc

// Copyright 2009-2011  Microsoft Corporation
//                2014  Johns Hopkins University (author: Daniel Povey)
//                2015  Vimal Manohar

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
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Apply log on a set of vectors in a Table (useful for probabilities)\n"
        "Usage: vector-apply-log [options] <in-rspecifier> <out-wspecifier>\n";

    bool invert = false;
    bool binary = false;

    ParseOptions po(usage);

    po.Register("invert", &invert, "Apply exp instead of log");
    po.Register("binary", &binary, "If true, write output as binary (only "
                "relevant for usage types two or three");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);
    
    if (ClassifyRspecifier(rspecifier, NULL, NULL) != kNoRspecifier) {
      BaseFloatVectorWriter vec_writer(wspecifier);

      SequentialBaseFloatVectorReader vec_reader(rspecifier);
      for (; !vec_reader.Done(); vec_reader.Next()) {
        Vector<BaseFloat> vec(vec_reader.Value());
        if (!invert)
          vec.ApplyLog();
        else 
          vec.ApplyExp();
        vec_writer.Write(vec_reader.Key(), vec);
      }
    } else {
      Vector<BaseFloat> vector;
      ReadKaldiObject(rspecifier, &vector);

      if (!invert)
        vector.ApplyLog();
      else 
        vector.ApplyExp();

      WriteKaldiObject(vector, wspecifier, binary);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



