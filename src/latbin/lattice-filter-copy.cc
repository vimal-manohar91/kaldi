// latbin/lattice-filter-copy.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2014  Vimal Manohar
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
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Copy lattices (e.g. useful for changing to text mode or changing\n"
        "format to standard from compact lattice.)\n"
        "Usage: lattice-filter-copy [options] filter_rxfilename lattice-rspecifier lattice-wspecifier\n"
        " e.g.: lattice-filter-copy --write-compact=false filter.scp ark:1.lats ark,t:text.lats\n"
        "See also: lattice-copy\n";
    
    ParseOptions po(usage);
    bool write_compact = true;
    po.Register("write-compact", &write_compact, "If true, write in normal (compact) form.");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string filter_rxfilename = po.GetArg(1),
        lats_rspecifier = po.GetArg(2),
        lats_wspecifier = po.GetArg(3);

    int32 n_done = 0, n_missing = 0;
    
    Input ki(filter_rxfilename);

    if (write_compact) {
      RandomAccessCompactLatticeReader lattice_reader(lats_rspecifier);
      CompactLatticeWriter lattice_writer(lats_wspecifier);
      
      std::string line;
      /* read each line from filter file */
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
        
        if (lattice_reader.HasKey(key)) {
          lattice_writer.Write(key, lattice_reader.Value(key));
          n_done++;
        } else {
          n_missing++;
        }
      }
      lattice_reader.Close();
    } else {
      RandomAccessLatticeReader lattice_reader(lats_rspecifier);
      LatticeWriter lattice_writer(lats_wspecifier);
      
      std::string line;
      /* read each line from filter file */
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
        
        if (lattice_reader.HasKey(key)) {
          lattice_writer.Write(key, lattice_reader.Value(key));
          n_done++;
        } else {
          n_missing++;
        }
      }
      lattice_reader.Close();
    }
 
    ki.Close();

    KALDI_LOG << "Done copying " << n_done << " lattices; missing " << n_missing << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

