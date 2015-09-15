// nnet2bin/nnet-extract-from-egs-discriminative.cc

// Copyright 2012-2013  Johns Hopkins University (author:  Daniel Povey)

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

namespace kaldi {
namespace nnet2 {
// returns an integer randomly drawn with expected value "expected_count"
// (will be either floor(expected_count) or ceil(expected_count)).
// this will go into an infinite loop if expected_count is very huge, but
// it should never be that huge.
int32 GetCount(double expected_count) {
  KALDI_ASSERT(expected_count >= 0.0);
  int32 ans = 0;
  while (expected_count > 1.0) {
    ans++;
    expected_count--;
  }
  if (WithProb(expected_count))
    ans++;
  return ans;
}
void AverageConstPart(int32 const_feat_dim,
                      DiscriminativeNnetExample *eg) {
  if (eg->spk_info.Dim() != 0) {  // already has const part.
    KALDI_ASSERT(eg->spk_info.Dim() == const_feat_dim);
    // and nothing to do.
  } else {
    int32 dim = eg->input_frames.NumCols(),
        basic_dim = dim - const_feat_dim;
    KALDI_ASSERT(const_feat_dim < eg->input_frames.NumCols());
    Matrix<BaseFloat> mat(eg->input_frames);  // copy to non-compressed matrix.
    eg->input_frames = mat.Range(0, mat.NumRows(), 0, basic_dim);
    eg->spk_info.Resize(const_feat_dim);
    eg->spk_info.AddRowSumMat(1.0 / mat.NumRows(),
                              mat.Range(0, mat.NumRows(),
                                        basic_dim, const_feat_dim),
                              0.0);
  }
}
                      

} // namespace nnet2
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Extract alignment and denominator lattice from examples for discriminative neural\n"
        "network training. \n"
        "\n"
        "Usage:  nnet-extract-from-egs-discriminative [options] <egs-rspecifier> <ali-wspecifier> <lat-wspecifier>\n"
        "\n"
        "e.g.\n"
        "nnet-extract-from-egs-discriminative ark:train.degs ark:ali ark:lat\n";
        
    std::string num_lat_wspecifier;
    
    ParseOptions po(usage);

    po.Register("num-lat-wspecifier", &num_lat_wspecifier, 
                "Extract numerator lattice if present");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1);
    std::string ali_wspecifier = po.GetArg(2);
    std::string lat_wspecifier = po.GetArg(3);

    SequentialDiscriminativeNnetExampleReader example_reader(
        examples_rspecifier);
    CompactLatticeWriter lattice_writer(lat_wspecifier);
    CompactLatticeWriter num_lattice_writer(num_lat_wspecifier);
    Int32VectorWriter alignment_writer(ali_wspecifier);

    int64 num_done = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_done++) {
      const std::string &key = example_reader.Key();
      const DiscriminativeNnetExample &eg = example_reader.Value();

      alignment_writer.Write(key, eg.num_ali);
      lattice_writer.Write(key, eg.den_lat);

      if (num_lat_wspecifier != "" && eg.num_lat_present) 
        num_lattice_writer.Write(key, eg.num_lat);
    }
    
    KALDI_LOG << "Extracted alignment and lattice from " << num_done << " discriminative neural-network training"
              << " examples";
    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



