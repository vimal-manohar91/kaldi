// bin/copy-vector.cc

// Copyright 2009-2012  Microsoft Corporation
//           2012-2014  Johns Hopkins University (Author: Daniel Povey)
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
#include "matrix/kaldi-vector.h"
#include "transform/transform-common.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
      "Down-weight vector dimensions that are lower than a threshold to "
      "specified weight. \n"
      "Optionally threshold up the weights that are higher than threshold\n"
      "\n"
      "Usage: thresh-vector [options] (<vector-in-rspecifier>|<vector-in-rxfilename>) (<vector-out-wspecifier>|<vector-out-wxfilename>)\n"
      " e.g.: thresh-vector --weight=0.7 1.mat -\n"
      "   copy-vector ark:2.trans ark,t:-\n"
      "see also: copy-vector, dot-weights, append-vector-to-feats\n";

    bool binary = true;
    BaseFloat threshold = 0.5;
    BaseFloat lower_cap = 0.0;
    BaseFloat upper_cap = -kLogZeroBaseFloat;

    ParseOptions po(usage);

    po.Register("binary", &binary, "Write in binary mode (only "
        "relevant if output is a wxfilename)");
    po.Register("threshold", &threshold, "Threshold below which we down-weight posteriors.");
    po.Register("lower-cap", &lower_cap, "Value to which we set values below threshold");
    po.Register("upper-cap", &upper_cap, "Value to which we set values above threshold");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string vector_in_fn = po.GetArg(1),
      vector_out_fn = po.GetArg(2);

    // all these "fn"'s are either rspecifiers or filenames.

    bool in_is_rspecifier =
      (ClassifyRspecifier(vector_in_fn, NULL, NULL)
       != kNoRspecifier),
      out_is_wspecifier =
        (ClassifyWspecifier(vector_out_fn, NULL, NULL, NULL)
         != kNoWspecifier);

    if (in_is_rspecifier != out_is_wspecifier)
      KALDI_ERR << "Cannot mix archives with regular files (copying vectors)";

    if (!in_is_rspecifier) {
      Vector<BaseFloat> vec;
      ReadKaldiObject(vector_in_fn, &vec);
      Output ko(vector_out_fn, binary);
      vec.LowerThreshold(threshold, lower_cap);
      if (upper_cap > kLogZeroBaseFloat)
        vec.UpperThreshold(threshold, upper_cap);
      vec.Write(ko.Stream(), binary);
      KALDI_LOG << "Copied vector to " << vector_out_fn;
      return 0;
    } else {
      int num_done = 0;
      BaseFloatVectorWriter writer(vector_out_fn);
      SequentialBaseFloatVectorReader reader(vector_in_fn);
      for (; !reader.Done(); reader.Next(), num_done++) {
        Vector<BaseFloat> vec (reader.Value());
        vec.LowerThreshold(threshold, lower_cap);
        if (upper_cap > kLogZeroBaseFloat)
          vec.UpperThreshold(threshold, upper_cap);
        writer.Write(reader.Key(), vec);
      }
      KALDI_LOG << "Thresholded " << num_done << " vectors";
      return (num_done != 0 ? 0 : 1);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



