// ivectorbin/sample-scores-into-trials.cc

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
#include "util/stl-utils.h"
#include "tree/clusterable-classes.h"
#include <unordered_map>

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Sample scores and convert them into trials\n"
      "Usage: sample-scores-into-trials [options] <scores-rspecifier> "
      "<reco2utt-rspecifier> <utt2spk-rspecifier> <trials-wxfilename>\n"
      "e.g.: \n"
      " sample-scores-into-trials ark:scores.ark ark,t:reco2utt ark,t:utt2spk -\n";

    ParseOptions po(usage);
    int32 num_target_trials = 50, num_nontarget_trials = 1000;

    po.Register("num-target-trials", &num_target_trials, 
                "The number of trials for target (same class) per input "
                "recording to write out");
    po.Register("num-nontarget-trials", &num_nontarget_trials, 
                "The number of trials for nontarget (different class) "
                "per input recording to write out");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string scores_rspecifier = po.GetArg(1),
      reco2utt_rspecifier = po.GetArg(2),
      utt2spk_rspecifier = po.GetArg(3),
      trials_wxfilename = po.GetArg(4);
    
    SequentialBaseFloatMatrixReader scores_reader(scores_rspecifier);
    RandomAccessTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
    SequentialTokenReader utt2spk_reader(utt2spk_rspecifier);

    Output ko(trials_wxfilename, false);

    GaussClusterable target_stats(1, 0.01);
    GaussClusterable nontarget_stats(1, 0.01);

    std::unordered_map<std::string, std::string, StringHasher> utt2spk;
    for (; !utt2spk_reader.Done(); utt2spk_reader.Next()) {
      utt2spk[utt2spk_reader.Key()] = utt2spk_reader.Value();
    }

    int32 num_err = 0, num_done = 0;
    for (; !scores_reader.Done(); scores_reader.Next()) {
      const std::string &reco = scores_reader.Key();
      const Matrix<BaseFloat> &scores(scores_reader.Value());

      if (!reco2utt_reader.HasKey(reco)) {
        KALDI_WARN << "Could not find uttlist for recording " << reco 
                   << " in " << reco2utt_rspecifier;
        num_err++;
        continue;
      }

      const std::vector<std::string> &uttlist = reco2utt_reader.Value(reco);
      std::vector<std::string> spklist;

      for (std::vector<std::string>::const_iterator it = uttlist.begin();
           it != uttlist.end(); ++it) {
        spklist.push_back(utt2spk[*it]);
      }

      int32 num_target_trials_found = 0, num_nontarget_trials_found = 0;
      while (num_nontarget_trials_found < num_nontarget_trials) {
        int32 id1 = RandInt(0, uttlist.size() - 1);
        int32 id2 = RandInt(0, uttlist.size() - 1);

        bool same_class = false;
        if (spklist[id1] == spklist[id2]) {
          same_class = true;
        }

        if (same_class) {
          if (num_target_trials_found < num_target_trials) {
            num_target_trials_found++;
            ko.Stream() << scores(id1, id2) << " " << "target\n";
            Vector<BaseFloat> vec(1);
            vec(0) = scores(id1, id2);
            target_stats.AddStats(vec);
          }
        } else {
          num_nontarget_trials_found++;
          ko.Stream() << scores(id1, id2) << " " << "nontarget\n";
          Vector<BaseFloat> vec(1);
          vec(0) = scores(id1, id2);
          nontarget_stats.AddStats(vec);
        }
      }

      int32 id = 0;
      while (num_target_trials_found < num_target_trials 
             && id < uttlist.size()) {
        num_target_trials_found++;
        ko.Stream() << scores(id, id) << " " << "target\n";
        Vector<BaseFloat> vec(1);
        vec(0) = scores(id, id);
        target_stats.AddStats(vec);
        id++;
      }

      num_done++;
    }

    double target_means = target_stats.x_stats()(0) / target_stats.Normalizer();
    double nontarget_means = nontarget_stats.x_stats()(0) 
                             / nontarget_stats.Normalizer();

    double target_vars = target_stats.x2_stats()(0) / target_stats.Normalizer()
                         - target_means * target_means;
    double nontarget_vars = nontarget_stats.x2_stats()(0) 
                            / nontarget_stats.Normalizer()
                            - nontarget_means * nontarget_means;

    KALDI_LOG << "(Means, Variances) are "
              << "(" << target_means << ", " << target_vars << ")"
              << "(" << nontarget_means << ", " << nontarget_vars << ")";
    KALDI_LOG << "mean-mid-score=" << (target_means + nontarget_means) / 2.0;
    KALDI_LOG << "optimum-threshold="
              << (target_means / target_vars 
                  + nontarget_means / nontarget_vars)
                 / (1.0 / target_vars + 1.0 / nontarget_vars);
    
    return (num_done > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

