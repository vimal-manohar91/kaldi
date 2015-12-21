// nnet3bin/discriminative-get-supervision.cc

// Copyright      2015  Johns Hopkins University (author:  Daniel Povey)
// Copyright 2014-2015  Vimal Manohar

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

#include <sstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "discriminative/discriminative-supervision.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get a 'sequence' supervision object for each file of training data.\n"
        "This will normally be piped into nnet3-discriminative-get-egs, where it\n"
        "will be split up into pieces and combined with the features.\n"
        "Input can come in two formats: \n"
        "from numerator alignments / denominator lattice pair \n"
        ", or from lattices\n"
        "(e.g. derived from aligning the data, see steps/align_fmllr_lats.sh)\n"
        "that have been converged to phone-level lattices with\n"
        "lattice-align-phones --replace-output-symbols=true.\n"
        "\n"
        "Usage: chain-get-supervision [options] <transition-model> <feature-specifier> <ali-rspecifier> \\\n" 
        "<den-lattice-rspecifier> <supervision-wspecifier>\n";

    bool lattice_input = false;
    DiscriminativeSupervisionOptions sup_opts;

    ParseOptions po(usage);
    sup_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string trans_model_rxfilename = po.GetArg(1),
                num_ali_rspecifier = po.GetArg(2),
                den_lat_rspecifier = po.GetArg(3),
                supervision_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    ReadKaldiObject(trans_model_rxfilename, &trans_model);

    DiscriminativeSupervisionWriter supervision_writer(supervision_wspecifier);
    RandomAccessCompactLatticeReader den_clat_reader(den_lat_rspecifier);
    SequentialInt32VectorReader ali_reader(ali_rspecifier);

    int32 num_utts_done = 0, num_utts_error = 0;

    for (; !ali_reader.Done(); ali_reader.Next())  {
      const std::string &key = ali_reader.Key();
      const std::vector<int32> &num_ali = ali_reader.Value();
      
      if (!den_clat_reader.HasKey(key)) {
        KALDI_WARN << "Could not find denominator lattice for utterance "
                   << key;
        num_utts_error++;
        continue;
      }

      if (!num_clat_reader.empty() && !num_clat_reader.HasKey(key)) {
        KALDI_WARN << "Could not find numerator lattice for utterance "
                   << key;
        num_utts_error++;
        continue;
      }

      const CompactLattice &den_clat = den_clat_reader.Value(key);

      DiscriminativeSupervision supervision;

      if (!num_clat_reader.empty()) {
        const CompactLattice &num_clat = num_clat_reader.Value(key);
        if (!LatticeToDiscriminativeSupervision(num_ali,
            num_clat, den_clat, 1.0, &supervision)) {
          KALDI_WARN << "Failed to convert lattice to supervision "
                     << "for utterance " << key;
          num_utts_error++;
          continue;
        }
      } else {
        if (!LatticeToDiscriminativeSupervision(num_ali,
            den_clat, 1.0, &supervision)) {
          KALDI_WARN << "Failed to convert lattice to supervision "
                     << "for utterance " << key;
          num_utts_error++;
          continue;
        }
      }

      supervision_writer.Write(key, supervision);
      
      num_utts_done++;
    } 
    
    KALDI_LOG << "Generated discriminative supervision information for "
              << num_utts_done << " utterances, errors on "
              << num_utts_error;
    return (num_utts_done > num_utts_error ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

