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
#include "nnet3/nnet-example-utils.h"
#include "lat/lattice-functions.h"

namespace kaldi {
namespace discriminative {

void UnitTestSupervisionSplitter(const SplitDiscriminativeSupervisionOptions &splitter_config,
                                 const DiscriminativeSupervision &supervision,
                                 const std::vector<int32> &range_starts, int32 frames_per_eg,
                                 std::vector<DiscriminativeSupervision*> *supervision_splits,
                                 Lattice *splitter_lat) {
  supervision_splits->clear();
  DiscriminativeSupervisionSplitter splitter(splitter_config, supervision);

  for (size_t i = 0; i < range_starts.size(); i++) {
    int32 range_start = range_starts[i];

    DiscriminativeSupervision* supervision_part = new DiscriminativeSupervision();

    splitter.GetFrameRange(range_start,
                           frames_per_eg,
                           supervision_part);

    supervision_splits->push_back(supervision_part);
      
    if (supervision.weights.size() > 0)
      KALDI_ASSERT(supervision_part->weights.size() > 0);

    for (size_t t = 0; t < frames_per_eg; t++) {
      KALDI_ASSERT(supervision_part->num_ali[t] == supervision.num_ali[t + range_start]);
      if (supervision.weights.size() > 0)
        KALDI_ASSERT(supervision_part->weights[t] == supervision.weights[t + range_start]);
      if (supervision.oracle_ali.size() > 0)
        KALDI_ASSERT(supervision_part->oracle_ali[t] == supervision.oracle_ali[t + range_start]);
    }
  }

  *splitter_lat = splitter.DenLat();
}

void UnitTestLatticeSplitPosteriors(const Lattice &lat, 
                               const std::vector<const Lattice*> &lat_splits,
                               const std::vector<int32> &range_starts,
                               std::vector<Posterior> *post_splits) {
  Posterior post;
  double lat_ac_like;
  std::vector<double> alpha;
  std::vector<double> beta;
  double lat_like = LatticeForwardBackward(lat, &post, &lat_ac_like, &alpha, &beta);
  KALDI_LOG << "Lattice score is " << lat_like;

  std::vector<double> alpha2;
  std::vector<double> beta2;
  ComputeLatticeAlphasAndBetas(lat, false, &alpha2, &beta2);

  KALDI_ASSERT(alpha == alpha2 && beta == beta2);

  std::vector<int32> state_times;
  LatticeStateTimes(lat, &state_times);

  post_splits->clear();
  post_splits->resize(lat_splits.size());

  size_t s = 0, n = 0;
  for (std::vector<const Lattice*>::const_iterator it = lat_splits.begin();
        it != lat_splits.end(); ++it, s++) {
    Posterior &post_part = (*post_splits)[s];
    std::vector<double> alpha_part;
    std::vector<double> beta_part;
    
    double lat_splits_ac_like = 0.0;
    double lat_splits_like = LatticeForwardBackward(**it, &post_part, &lat_splits_ac_like, &alpha_part, &beta_part);

    if (lat_splits_like > 1e-7) 
      KALDI_WARN << "lat_splits_like = " << lat_splits_like << " is greater than 1e-7";
    
    n = std::lower_bound(state_times.begin(), state_times.end(), range_starts[s]) - state_times.begin();
    if (s == 0) n--;
    //for (size_t n_part = 1; n_part < alpha_part.size()-1; n_part++) {
    //  KALDI_ASSERT(kaldi::ApproxEqual(alpha_part[n_part], alpha[n_part + n], .1));
    //}

    for (size_t i = 0; i < post_part.size(); i++) {
      size_t t = i + range_starts[s];
      KALDI_ASSERT(post_part[i].size() == post[t].size());
      for (size_t j = 0; j < post_part[i].size(); j++) {
        KALDI_ASSERT(post_part[i][j].first == post[t][j].first);
        if (post_part[i][j].second < 1e-6 && post[t][j].second < 1e-6) continue;
        //KALDI_ASSERT(kaldi::ApproxEqual(post_part[i][j].second, post[t][j].second, .1) || std::abs(post_part[i][j].second - post[t][j].second) < .1);
      }
    }
  }
  //KALDI_ASSERT(kaldi::ApproxEqual(lat_ac_like, lat_splits_ac_like));
  //KALDI_ASSERT(kaldi::ApproxEqual(lat_like, lat_splits_like));
}

void UnitTestMmiPosteriors(
    const TransitionModel &tmodel,
    const std::vector<int32> &num_ali,
    const Lattice &lat, 
    const std::vector<const std::vector<int32>*> &ali_splits,
    const std::vector<const Lattice*> &lat_splits, 
    const std::vector<int32> range_starts,
    std::vector<Posterior> *post_splits) {
  Posterior post;
  LatticeForwardBackwardMmi(tmodel, lat, num_ali, 
                            false, false, false, &post);

  size_t s = 0;
  for (size_t i = 0; i < ali_splits.size(); i++, s++) {
    Posterior post_part;
    LatticeForwardBackwardMmi(tmodel, *(lat_splits[i]), 
                              *(ali_splits[i]), false, false, false,
                              &post_part);
    post_splits->push_back(post_part);
    
    for (size_t i = 0; i < post_part.size(); i++) {
      size_t t = i + range_starts[s];
      KALDI_ASSERT(post_part[i].size() == post[t].size());
      for (size_t j = 0; j < post_part[i].size(); j++) {
        KALDI_ASSERT(post_part[i][j].first == post[t][j].first);
        if (post_part[i][j].second < 1e-6 && post[t][j].second < 1e-6) continue;
        KALDI_ASSERT(kaldi::ApproxEqual(post_part[i][j].second, post[t][j].second, .1) || std::abs(post_part[i][j].second - post[t][j].second) < .1);
      }
    }
  }
}

void UnitTestMpePosteriors(
    const TransitionModel &tmodel,
    const std::vector<int32> &silence_phones,
    std::string criterion,
    bool one_silence_class,
    const std::vector<int32> &num_ali,
    const Lattice &lat, 
    const std::vector<const std::vector<int32>*> &ali_splits,
    const std::vector<const Lattice*> &lat_splits, 
    const std::vector<int32> range_starts,
    std::vector<Posterior> *post_splits) {
  Posterior post;
  
  double acc;
  try {
    acc = LatticeForwardBackwardMpeVariants(tmodel, silence_phones, lat, num_ali, 
                                      criterion, one_silence_class, &post);
  } catch (std::exception &e) {
    KALDI_LOG << e.what();
    return;
  }

  size_t s = 0;
  double splits_acc = 0;
  int32 splits_count = 0;
  for (size_t i = 0; i < ali_splits.size(); i++, s++) {
    Posterior post_part;
    try {
      splits_acc += LatticeForwardBackwardMpeVariants(tmodel, silence_phones, *(lat_splits[i]), 
          *(ali_splits[i]), criterion, one_silence_class,
          &post_part);
      splits_count += (ali_splits[i])->size();
    } catch (std::exception &e) {
      KALDI_LOG << e.what();
      continue;
    }
    post_splits->push_back(post_part);
    
    for (size_t i = 0; i < post_part.size(); i++) {
      size_t t = i + range_starts[s];

      KALDI_ASSERT(post_part[i].size() == 0 || post[t].size() == 0 || post_part[i].size() == post[t].size());
      for (size_t j = 0; j < post_part[i].size(); j++) {
        KALDI_ASSERT(post_part[i][j].first == post[t][j].first);
        if (i < 10 || i > post_part.size() - 10) continue;
        if (post_part[i][j].second < 1e-6 && post[t][j].second < 1e-6) continue;
        if(!(kaldi::ApproxEqual(post_part[i][j].second, post[t][j].second, .1) || std::abs(post_part[i][j].second - post[t][j].second) < .1)) {
          KALDI_WARN << "MPE split post = " << post_part[i][j].second << " vs " << post[t][j].second;
          if (i > 20 && i < post_part.size() - 20) 
            KALDI_ASSERT(kaldi::ApproxEqual(post_part[i][j].second, post[t][j].second, .1) || std::abs(post_part[i][j].second - post[t][j].second) < .1);
        }
      }
    }
  }

  if(!kaldi::ApproxEqual(acc / num_ali.size(), splits_acc / splits_count, 1e-2)) {
    KALDI_ASSERT(kaldi::ApproxEqual(acc / num_ali.size(), splits_acc / splits_count, 1e-1));
    KALDI_WARN << "acc = " << acc / num_ali.size() << ", while splits acc = " << splits_acc / splits_count;
  }
}

void UnitTestSupervisionMerge(const DiscriminativeSupervision &supervision,
                              const std::vector<const DiscriminativeSupervision*> &supervision_splits,
                              const std::vector<int32> &range_starts,
                              const std::vector<Posterior> &post_splits) {
  std::vector<DiscriminativeSupervision> out_supervisions;
  AppendSupervision(supervision_splits, true, &out_supervisions);
  DiscriminativeSupervision &out_supervision = out_supervisions.back();

  Posterior post, out_post;
  LatticeForwardBackward(supervision.den_lat, &post, NULL);
  LatticeForwardBackward(out_supervision.den_lat, &out_post, NULL);
   
  for (size_t s = 0; s < range_starts.size(); s++) {
    for (size_t i = 0; i < out_supervision.frames_per_sequence; i++) {
      size_t t = range_starts[s] + i, ot = s * out_supervision.frames_per_sequence + i;
      KALDI_ASSERT(out_post[ot].size() == post[t].size());
      for (size_t j = 0; j < out_post[ot].size(); j++) {
        KALDI_ASSERT(out_post[ot][j].first == post[t][j].first);
        if (out_post[ot][j].second < 1e-6 && post[t][j].second < 1e-6) continue;
        //KALDI_ASSERT(kaldi::ApproxEqual(out_post[ot][j].second, post[t][j].second, .1) || std::abs(out_post[ot][j].second - post[t][j].second) < 1e-4);
      }
    }
  }
}

void UnitTestSupervision(const SplitDiscriminativeSupervisionOptions &splitter_config,
                         const TransitionModel &tmodel,
                         DiscriminativeSupervision *supervision,
                         int32 frames_per_eg) {
  int32 num_frames = supervision->frames_per_sequence;
  KALDI_ASSERT(supervision->num_sequences == 1);

  std::vector<int32> range_starts;
  nnet3::SplitIntoRanges(num_frames, frames_per_eg, &range_starts);

  KALDI_ASSERT(!range_starts.empty());

  std::vector<DiscriminativeSupervision*> supervision_splits;

  Lattice splitter_lat;
  UnitTestSupervisionSplitter(splitter_config, *supervision, range_starts, frames_per_eg, &supervision_splits, &splitter_lat);

  std::vector<const Lattice*> lat_splits;
  std::vector<const std::vector<int32>*> ali_splits;
  for (std::vector<DiscriminativeSupervision*>::iterator it = supervision_splits.begin(); 
        it != supervision_splits.end(); ++it) {
    fst::ScaleLattice(fst::AcousticLatticeScale(splitter_config.supervision_config.acoustic_scale), &((*it)->den_lat));
    lat_splits.push_back(const_cast<Lattice*>(&((*it)->den_lat)));
    ali_splits.push_back(const_cast<std::vector<int32>*>(&((*it)->num_ali)));
  }

  supervision->den_lat = splitter_lat;
  Lattice &den_lat = supervision->den_lat;
 
  // Check that the posteriors for all the frames
  // before and after splitting
  std::vector<Posterior> post_splits;
  UnitTestLatticeSplitPosteriors(den_lat, lat_splits, range_starts, &post_splits);

  std::vector<Posterior> mmi_post_splits;
  UnitTestMmiPosteriors(tmodel,
                        supervision->num_ali, supervision->den_lat,
                        ali_splits, lat_splits, 
                        range_starts, &mmi_post_splits);
 
  std::vector<int32> silence_phones;
  for (int32 i = 1; i <= 15; i++) silence_phones.push_back(i);

  std::string criterion = "smbr";
  bool one_silence_class = true;
  for (int32 i = 0; i < 4; i++) {
    if (i == 1) one_silence_class = false;
    if (i == 2) criterion = "mpfe";
    if (i == 3) one_silence_class = true;

    std::vector<Posterior> mpe_post_splits;
    UnitTestMpePosteriors(tmodel, silence_phones, criterion, one_silence_class, 
                          supervision->num_ali, supervision->den_lat,
                          ali_splits, lat_splits, 
                          range_starts, &mpe_post_splits);
  }

  std::vector<const DiscriminativeSupervision*> supervision_splits_const;
  for (size_t i = 0; i < supervision_splits.size(); i++) {
    supervision_splits_const.push_back(const_cast<DiscriminativeSupervision*> (supervision_splits[i]));
  }

  // Check again after merging
  UnitTestSupervisionMerge(*supervision, supervision_splits_const, range_starts, post_splits);
  
  for (std::vector<DiscriminativeSupervision*>::iterator it = supervision_splits.begin(); it != supervision_splits.end(); ++it) {
    delete *it;
  }
}

}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::discriminative;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get a discriminative training supervision object for each file of training data\n"
        " and test if splitting and merging works correctly\n"
        "Input can come in two formats: \n"
        "numerator alignments / denominator lattice pair \n"
        ", or numerator and denominator lattice pair\n"
        "Usage: discriminative-get-supervision [options] <ali-rspecifier> \\\n" 
        "<den-lattice-rspecifier>\n";

    std::string num_lat_rspecifier;
    std::string oracle_rspecifier;
    std::string frame_weights_rspecifier;
    int32 frames_per_eg = 150;

    discriminative::SplitDiscriminativeSupervisionOptions splitter_config;

    ParseOptions po(usage);
    po.Register("num-lat-rspecifier", &num_lat_rspecifier, "Get supervision "
                "with numerator lattice");
    po.Register("oracle-rspecifier", &oracle_rspecifier, "Add oracle "
                "alignment to supervision");
    po.Register("frame-weights-rspecifier", &frame_weights_rspecifier,
                "Add frame weights to supervision");
    po.Register("num-frames", &frames_per_eg, "Number of frames with labels "
                "that each example contains.  Will be rounded up to a multiple "
                "of --frame-subsampling-factor.");
    
    ParseOptions splitter_opts("supervision-splitter", &po);
    splitter_config.Register(&splitter_opts);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string model_rxfilename = po.GetArg(1),
                num_ali_rspecifier = po.GetArg(2),
                den_lat_rspecifier = po.GetArg(3);

    TransitionModel tmodel;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      tmodel.Read(ki.Stream(), binary);
    }

    RandomAccessCompactLatticeReader den_lat_reader(den_lat_rspecifier);
    SequentialInt32VectorReader ali_reader(num_ali_rspecifier);

    RandomAccessCompactLatticeReader num_lat_reader(num_lat_rspecifier);
    RandomAccessInt32VectorReader oracle_reader(oracle_rspecifier);
    RandomAccessBaseFloatVectorReader frame_weights_reader(frame_weights_rspecifier);

    int32 num_utts_done = 0, num_utts_error = 0;

    for (; !ali_reader.Done(); ali_reader.Next())  {
      const std::string &key = ali_reader.Key();
      const std::vector<int32> &num_ali = ali_reader.Value();
      
      if (!den_lat_reader.HasKey(key)) {
        KALDI_WARN << "Could not find denominator lattice for utterance "
                   << key;
        num_utts_error++;
        continue;
      }

      if (!num_lat_rspecifier.empty() && !num_lat_reader.HasKey(key)) {
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

      const CompactLattice &den_clat = den_lat_reader.Value(key);

      DiscriminativeSupervision supervision;

      if (!num_lat_rspecifier.empty()) {
        const CompactLattice &num_clat = num_lat_reader.Value(key);
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

      if (supervision.frames_per_sequence < frames_per_eg) continue;

      UnitTestSupervision(splitter_config, tmodel, &supervision, frames_per_eg);

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


