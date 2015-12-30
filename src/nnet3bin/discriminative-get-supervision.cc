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
#include "nnet3/discriminative-supervision.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::discriminative;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get a discriminative training supervision object for each file of training data.\n"
        "This will normally be piped into nnet3-discriminative-get-egs, where it\n"
        "will be split up into pieces and combined with the features.\n"
        "Input can come in two formats: \n"
        "numerator alignments / denominator lattice pair \n"
        ", or numerator and denominator lattice pair\n"
        "Usage: discriminative-get-supervision [options] <ali-rspecifier> \\\n" 
        "<den-lattice-rspecifier> <supervision-wspecifier>\n";

    std::string num_lat_rspecifier;
    std::string oracle_rspecifier;
    std::string frame_weights_rspecifier;

    DiscriminativeSupervisionOptions sup_opts;

    ParseOptions po(usage);
    po.Register("num-lat-rspecifier", &num_lat_rspecifier, "Get supervision "
                "with numerator lattice");
    po.Register("oracle-rspecifier", &oracle_rspecifier, "Add oracle "
                "alignment to supervision");
    po.Register("frame-weights-rspecifier", &frame_weights_rspecifier,
                "Add frame weights to supervision");

    sup_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string num_ali_rspecifier = po.GetArg(1),
                den_lat_rspecifier = po.GetArg(2),
                supervision_wspecifier = po.GetArg(3);

    DiscriminativeSupervisionWriter supervision_writer(supervision_wspecifier);
    RandomAccessCompactLatticeReader den_clat_reader(den_lat_rspecifier);
    SequentialInt32VectorReader ali_reader(num_ali_rspecifier);

    RandomAccessCompactLatticeReader num_clat_reader(num_lat_rspecifier);
    RandomAccessInt32VectorReader oracle_reader(oracle_rspecifier);
    RandomAccessBaseFloatVectorReader frame_weights_reader(frame_weights_rspecifier);

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

      if (!num_lat_rspecifier.empty() && !num_clat_reader.HasKey(key)) {
        KALDI_WARN << "Could not find numerator lattice for utterance "
                   << key;
        num_utts_error++;
        continue;
      }
      
      if (!oracle_rspecifier.empty() && !oracle_reader.HasKey(key)) {
        KALDI_WARN << "Could not find oracle alignment for utterance "
                   << key;
        num_utts_error++;
        continue;
      }

      if (!frame_weights_rspecifier.empty() && !frame_weights_reader.HasKey(key)) {
        KALDI_WARN << "Could not find frame weights for utterance "
                   << key;
        num_utts_error++;
        continue;
      }

      Vector<BaseFloat> frame_weights;
      std::vector<int32> oracle_ali;
      
      if (!oracle_rspecifier.empty()) {
        oracle_ali = oracle_reader.Value(key);
      }

      if (!frame_weights_rspecifier.empty()) {
        frame_weights = frame_weights_reader.Value(key);
      }

      const CompactLattice &den_clat = den_clat_reader.Value(key);

      DiscriminativeSupervision supervision;

      if (!num_lat_rspecifier.empty()) {
        const CompactLattice &num_clat = num_clat_reader.Value(key);
        if (!LatticeToDiscriminativeSupervision(num_ali,
            num_clat, den_clat, 1.0, &supervision, 
            (!frame_weights_rspecifier.empty() ? &frame_weights : NULL), 
            (!oracle_rspecifier.empty() ? &oracle_ali : NULL))) {
          KALDI_WARN << "Failed to convert lattice to supervision "
                     << "for utterance " << key;
          num_utts_error++;
          continue;
        }
      } else {
        if (!LatticeToDiscriminativeSupervision(num_ali,
            den_clat, 1.0, &supervision,
            (!frame_weights_rspecifier.empty() ? &frame_weights : NULL), 
            (!oracle_rspecifier.empty() ? &oracle_ali : NULL))) {
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

