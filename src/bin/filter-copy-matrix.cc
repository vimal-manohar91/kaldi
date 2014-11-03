// bin/filter-copy-matrix.cc

// Copyright 2009-2011  Microsoft Corporation
// Copyright      2014  Vimal Manohar

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
#include "transform/transform-common.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Filter archives of matrices (e.g. features or transforms) "
        "copy only those in filter list"
        "Also see copy-matrix\n"
        "\n"
        "Usage: filter-copy-matrix [options] <filter-wxfilename> <matrix-in-rspecifier> <matrix-out-wspecifier>\n"
        " e.g.: filter-copy-matrix filter.scp ark:2.trans ark,t:-\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }


    std::string filter_rxfilename = po.GetArg(1),
        matrix_rspecifier = po.GetArg(2),
        matrix_wspecifier = po.GetArg(3);

    int n_done = 0, n_missing = 0;

    BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);
    RandomAccessBaseFloatMatrixReader matrix_reader(matrix_rspecifier);

    Input ki(filter_rxfilename);

    std::string line;
    while (std::getline(ki.Stream(), line)) {
      std::vector<std::string> split_line;
      // Split the line by space or tab and check the number of fields in each
      // line. The file must have atleast one field and the first one is taken
      // as utterance id.
      SplitStringToVector(line, " \t\r", true, &split_line);
      if (split_line.size() < 1) {
        KALDI_WARN << "Invalid line in segments file: " << line;
        continue;
      }

      std::string key = split_line[0];
       
      if (matrix_reader.HasKey(key)) {
        matrix_writer.Write(key, matrix_reader.Value(key));
        n_done++;
      } else {
        n_missing++;
      }
    }
 
    ki.Close();

    KALDI_LOG << "Copied " << n_done << " matrices.";
    return (n_done > n_missing ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

