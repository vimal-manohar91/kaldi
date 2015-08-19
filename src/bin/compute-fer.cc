// bin/compute-fer.cc

// Copyright 2009-2011  Microsoft Corporation
//                2014  Johns Hopkins University (authors: Jan Trmal, Daniel Povey)
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
#include "util/parse-options.h"
#include "tree/context-dep.h"

namespace kaldi {

template<typename T>
void PrintAlignmentStats(const std::vector<T> &ref,
                         const std::vector<T> &hyp,
                         std::ostream &os) {

  for (size_t i = 0; i < ref.size(); i++) {
    if (ref[i] != hyp[i]) {
      os << "substitution " << ref[i] << ' ' << hyp[i] << '\n';
    } else {
      os << "correct" << ref[i] << '\n';
    }
  }
}

}


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Compute frame-error rate between two alignments\n"
        "Takes two alignment files, in integer or text format,\n"
        "and outputs overall frame-error rate statistics to standard output.\n"
        "Optionally, the third argument can be used to obtain detailed statistics\n"
        "\n"
        "Usage: compute-fer [options] <ref-rspecifier> <hyp-rspecifier> [<stats-out>]\n"
        "\n"
        "E.g.: compute-fer --text --mode=present ark:data/train/text ark:hyp_text\n"
        "or: compute-fer --text --mode=present ark:data/train/text ark:hyp_text - | \\\n"
        "   sort | uniq -c\n";

    ParseOptions po(usage);

    std::string mode = "strict";
    bool text_input = false;  //  if this is true, we expect symbols as strings,

    po.Register("mode", &mode,
                "Scoring mode: \"present\"|\"all\"|\"strict\":\n"
                "  \"present\" means score those we have transcriptions for\n"
                "  \"all\" means treat absent transcriptions as empty\n"
                "  \"strict\" means die if all in ref not also in hyp");
    po.Register("text", &text_input, "Expect strings, not integers, as input.");

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string ref_rspecifier = po.GetArg(1);
    std::string hyp_rspecifier = po.GetArg(2);

    Output stats_output;
    bool detailed_stats = (po.NumArgs() == 3);
    if (detailed_stats)
      stats_output.Open(po.GetOptArg(3), false, false);  // non-binary output
    
    if (mode != "strict" && mode != "present" && mode != "all") {
      KALDI_ERR << "--mode option invalid: expected \"present\"|\"all\"|\"strict\", got "
                << mode;
    }

    int64 num_words = 0, num_sent = 0, sent_errs = 0,
          num_sub = 0, num_absent_sents = 0, num_mismatch_sents = 0;

    {
      SequentialInt32VectorReader ref_reader(ref_rspecifier);
      RandomAccessInt32VectorReader hyp_reader(hyp_rspecifier);

      for (; !ref_reader.Done(); ref_reader.Next()) {
        std::string key = ref_reader.Key();
        const std::vector<int32> &ref_sent = ref_reader.Value();
        std::vector<int32> hyp_sent;
        if (!hyp_reader.HasKey(key)) {
          if (mode == "strict")
            KALDI_ERR << "No hypothesis for key " << key << " and strict "
                "mode specifier.";
          num_absent_sents++;
          if (mode == "present")  // do not score this one.
            continue;
        } else {
          hyp_sent = hyp_reader.Value(key);
        }

        if (hyp_sent.size() != ref_sent.size()) {
          if (mode == "strict") {
            KALDI_ERR << "Reference and hypothesis are of different lengths "
                      << "for key " << key << " and strict mode specifier.";
          }
          num_mismatch_sents++;
          if (mode == "present")  // do not score this one
            continue;
        }

        int64 sub = 0;
        for (size_t i = 0; i < ref_sent.size(); i++) {
          if (ref_sent[i] != hyp_sent[i]) sub++;
        }
        num_sub += sub;
        num_words += ref_sent.size();

        if (detailed_stats) {
          PrintAlignmentStats(ref_sent, hyp_sent, stats_output.Stream());
        }
        num_sent++;
        sent_errs += (ref_sent != hyp_sent);
      }
    } 
    //else {
    //  SequentialTokenVectorReader ref_reader(ref_rspecifier);
    //  RandomAccessTokenVectorReader hyp_reader(hyp_rspecifier);

    //  for (; !ref_reader.Done(); ref_reader.Next()) {
    //    std::string key = ref_reader.Key();
    //    const std::vector<std::string> &ref_sent = ref_reader.Value();
    //    std::vector<std::string> hyp_sent;
    //    if (!hyp_reader.HasKey(key)) {
    //      if (mode == "strict")
    //        KALDI_ERR << "No hypothesis for key " << key << " and strict "
    //            "mode specifier.";
    //      num_absent_sents++;
    //      if (mode == "present")  // do not score this one.
    //        continue;
    //    } else {
    //      hyp_sent = hyp_reader.Value(key);
    //    }
    //    int32 ins, del, sub;
    //    int32 this_word_errs = LevenshteinEditDistance(ref_sent, hyp_sent,
    //                                         &ins, &del, &sub);

    //    word_errs += this_word_errs;

    //    int32 this_num_words = ref_sent.size();
    //    num_words += this_num_words;

    //    if (GetVerboseLevel() > 1) {
    //      BaseFloat percent_wer = 100.0 * static_cast<BaseFloat>(this_word_errs)
    //          / static_cast<BaseFloat>(ref_sent.size());
    //      std::cerr.precision(2);
    //      std::cerr << key << " %WER " << std::fixed << percent_wer << " [ " << this_word_errs
    //          << " / " << this_num_words << ", " << ins << " ins, "
    //          << del << " del, " << sub << " sub ]" << '\n';
    //    }


    //    num_ins += ins;
    //    num_del += del;
    //    num_sub += sub;

    //    if (detailed_stats) {
    //      const std::string eps = "";
    //      PrintAlignmentStats(ref_sent, hyp_sent, eps, stats_output.Stream());
    //    }
    //    num_sent++;
    //    sent_errs += (ref_sent != hyp_sent);
    //  }
    //}

    BaseFloat percent_wer = 100.0 * static_cast<BaseFloat>(num_sub)
        / static_cast<BaseFloat>(num_words);
    std::cout.precision(2);
    std::cerr.precision(2);
    std::cout << "%WER " << std::fixed << percent_wer << " [ " << num_sub
              << " / " << num_words << ", " << 0 << " ins, "
              << 0 << " del, " << num_sub << " sub ]"
              << (num_absent_sents != 0 ? " [PARTIAL]" : "") << '\n';
    BaseFloat percent_ser = 100.0 * static_cast<BaseFloat>(sent_errs)
        / static_cast<BaseFloat>(num_sent);
    std::cout << "%SER " << std::fixed << percent_ser <<  " [ "
               << sent_errs << " / " << num_sent << " ]\n";
    std::cout << "Scored " << num_sent << " sentences, "
              << num_absent_sents << " not present in hyp.\n";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

