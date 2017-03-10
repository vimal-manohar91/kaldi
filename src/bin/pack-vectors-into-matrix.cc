// bin/pack-vectors-into-matrix.cc

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
        "Pack vectors indexed by say, utterance ids, into matrices "
        "indexes by, say speaker ids.\n"
        "\n"
        "Usage: pack-vectors-into-matrix [options] <vector-in-rspecifier> "
        "<spk2utt> <matrix-out-wspecifier>\n"
        " e.g.: pack-vectors-into-matrix ark:vec.ark ark,t:spk2utt ark:mat.ark\n"
        "See also: unpack-matrix-into-vectors, copy-matrix, copy-feats\n";

    ParseOptions po(usage);

    bool ignore_missing = false;

    po.Register("ignore-missing", &ignore_missing,
                "If true, missing vectors are ignored while "
                "packing into matrix.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string vector_rspecifier = po.GetArg(1),
        spk2utt_rspecifier = po.GetArg(2),
        matrix_wspecifier = po.GetArg(3);

    int32 num_spk = 0, num_utts = 0, num_utt_err = 0, num_spk_err = 0;
    BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);
    RandomAccessBaseFloatVectorReader vector_reader(vector_rspecifier);
    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);

    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      const std::string &spk = spk2utt_reader.Key();
      const std::vector<std::string> &utts = spk2utt_reader.Value();

      bool flag = true;
      Matrix<BaseFloat> matrix;

      int32 id = 0;
      for (size_t i = 0; i < utts.size(); i++) {
        if (!vector_reader.HasKey(utts[i])) {
          KALDI_WARN << "Could not find vector for utterance " << utts[i]
                     << " in rspecifier " << vector_rspecifier;
          if (ignore_missing) {
            num_utt_err++;
            continue;
          } else {
            flag = false;
            break;
          }
        }
        num_utts++;

        const Vector<BaseFloat> &vector = vector_reader.Value(utts[i]);
        if (matrix.NumRows() == 0) {
          matrix.Resize(utts.size(), vector.Dim());
        }
        matrix.CopyRowFromVec(vector, id++);
      }

      if (!flag || id == 0) {
        num_spk_err++;
        continue;
      }

      matrix.Resize(id, matrix.NumCols(), kCopyData);
      matrix_writer.Write(spk, matrix);
      num_spk++;
    } 
    
    KALDI_LOG << "Packed " << num_utts << " vectors into " 
              << num_spk << " matrices; missing " << num_utt_err 
              << " vectors and failed with " << num_spk_err << " matrices.";
    return (num_utts != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
