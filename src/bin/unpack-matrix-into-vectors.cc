// bin/unpack-matrix-into-vectors.cc

// Copyright 2017  Vimal Manohar

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
        "Unpack matrices into vectors indexes by unique ids.\n"
        "\n"
        "Usage: unpack-matrix-into-vectors [options] <matrix-in-rspecifier> "
        "<vector-out-wspecifier> <out-spk2utt>\n"
        " e.g.: unpack-matrix-into-vectors ark:mat.ark ark:vec.ark ark,t:spk2utt\n"
        "See also: pack-vectors-into-matrix, copy-matrix, copy-feats\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string matrix_rspecifier = po.GetArg(1),
        vector_wspecifier = po.GetArg(2),
        spk2utt_wspecifier = po.GetArg(3);

    int32 num_spk = 0, num_utts = 0, num_spk_err = 0;
    SequentialBaseFloatMatrixReader matrix_reader(matrix_rspecifier);
    BaseFloatVectorWriter vector_writer(vector_wspecifier);
    TokenVectorWriter spk2utt_writer(spk2utt_wspecifier);

    for (; !matrix_reader.Done(); matrix_reader.Next()) {
      const std::string &spk = matrix_reader.Key();
      const Matrix<BaseFloat> &matrix = matrix_reader.Value();

      if (matrix.NumRows() == 0) {
        num_spk_err++;
        continue;
      }

      std::vector<std::string> utts;
      for (int32 i = 0; i < matrix.NumRows(); i++) {
        std::ostringstream oss;
        oss << spk << "-" << i;
        utts.push_back(oss.str());
        vector_writer.Write(utts[i], Vector<BaseFloat>(matrix.Row(i)));
        num_utts++;
      }
      spk2utt_writer.Write(spk, utts);

      num_spk++;
    } 
    
    KALDI_LOG << "Unpacked " << num_spk << " matrices into " 
              << num_utts << " vectors; failed with "
              << num_spk_err << " matrices.";
    return (num_spk != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
