// chainbin/nnet3-chain-copy-egs-and-perturb.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
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
#include "hmm/transition-model.h"
#include "nnet3/nnet-chain-example.h"
#include "feat/signal.h"
#include "feat/resample.h"
//#include <sox.h>
namespace kaldi {
namespace nnet3 {
// returns an integer randomly drawn with expected value "expected_count"
// (will be either floor(expected_count) or ceil(expected_count)).
int32 GetCount(double expected_count) {
  KALDI_ASSERT(expected_count >= 0.0);
  int32 ans = floor(expected_count);
  expected_count -= ans;
  if (WithProb(expected_count))
    ans++;
  return ans;
}

void PerturbRawChainExample(BaseFloat max_rand_shift, 
                            BaseFloat max_speed_perturb,
                            const Vector<BaseFloat> *speaker_own_filter,
                            const Vector<BaseFloat> *speaker_inv_filter,
                            NnetChainExample *eg) {
  Matrix<BaseFloat> feat_mat;
  eg->inputs[0].features.GetMatrix(&feat_mat);
  int32 max_shift = int(max_rand_shift * feat_mat.NumCols());
  int32 orig_size = feat_mat.NumRows() * feat_mat.NumCols(), 
    row_size = feat_mat.NumCols();
  Vector<BaseFloat> feat_vec(orig_size);
  // Vectorize the matrix
  for (int32 row = 0; row < feat_mat.NumRows(); row++) 
    feat_vec.Range(row * row_size, row_size).CopyFromVec(feat_mat.Row(row));
  // choose a sample shift randomly and perturb egs by sample shift.
  Vector<BaseFloat> shifted_feat(feat_vec);
  if (max_rand_shift != 0) {
    int32 shifted_size = (feat_mat.NumRows()-1) * feat_mat.NumCols(),
      rand_shift = RandInt(0, max_shift);
    shifted_feat.Resize(shifted_size);
    shifted_feat.CopyFromVec(feat_vec.Range(rand_shift, shifted_size));
  }
  
  Vector<BaseFloat> stretched_feat(shifted_feat);
  // If nonzero, apply speed perturbation on raw-egs using sox with randomly 
  // generated speed perturabation valu in [1 - max_speed_perturb 1 + max_speed_perturb]
  if (max_speed_perturb != 0) {
    KALDI_ASSERT(max_speed_perturb < 1);
    int32 input_dim = orig_size, output_dim = orig_size - feat_mat.NumCols();
    Vector<BaseFloat> samp_points_secs(orig_size);
    BaseFloat samp_freq = 2000, min_stretch = 0.0, 
      max_stretch = max_speed_perturb;
    // we stretch the middle part of the spliced wave and the input should be expanded
    // by extra frame to be larger than the output length => s * (m+n)/2 < m.
    // y((m - n + 2 * t)/2) = x(s * (m - n + 2 * t)/2) for t = 0,..,n 
    KALDI_ASSERT(input_dim > output_dim * ((1.0 + max_stretch) / (1.0 - max_stretch)));
    // Generate random stretch value between -max_stretch, max_stretch.
    int32 max_stretch_int = static_cast<int32>(max_stretch * 1000);
    BaseFloat stretch = static_cast<BaseFloat>(RandInt(-max_stretch_int, max_stretch_int) / 1000.0); 
    if (abs(stretch) > min_stretch) {
      int32 num_zeros = 4; // Number of zeros of the sinc function that the window extends out to.
      BaseFloat filter_cutoff_hz = samp_freq * 0.475; // lowpass frequency that's lower than 95% of 
                                                      // the Nyquist.
      for (int32 i = 0; i < output_dim; i++) 
        samp_points_secs(i) = static_cast<BaseFloat>(((1.0 + stretch) * 
          (0.5 * (input_dim - output_dim) + i))/ samp_freq);

      ArbitraryResample time_resample(input_dim, samp_freq,
                                      filter_cutoff_hz, 
                                      samp_points_secs,
                                      num_zeros);
      time_resample.Resample(shifted_feat, &stretched_feat);
    }
  }

  if (speaker_own_filter) {
    KALDI_ASSERT(speaker_own_filter->Dim() != 0);
    FFTbasedBlockConvolveSignals((*speaker_own_filter), &stretched_feat, false);  
  }

  if (speaker_inv_filter) {
    KALDI_ASSERT(speaker_inv_filter->Dim() != 0);
    FFTbasedBlockConvolveSignals((*speaker_inv_filter), &stretched_feat, true);  
  }

  stretched_feat.Resize(orig_size, kCopyData);
  
  // Unvectorize vec to mat
  for (int32 row = 0; row < feat_mat.NumRows(); row++)
    feat_mat.CopyRowFromVec(stretched_feat.Range(row * row_size, row_size), row);
  eg->inputs[0].features = feat_mat; 
}

} // namespace nnet3
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy examples for nnet3+chain network training, possibly changing the binary mode.\n"
        "Supports multiple wspecifiers, in which case it will write the examples\n"
        "round-robin to the outputs.\n"
        "\n"
        "Usage:  nnet3-chain-copy-egs [options] <egs-rspecifier> <egs-wspecifier1> [<egs-wspecifier2> ...]\n"
        "\n"
        "e.g.\n"
        "nnet3-chain-copy-egs ark:train.cegs ark,t:text.cegs\n"
        "or:\n"
        "nnet3-chain-copy-egs ark:train.cegs ark:1.cegs ark:2.cegs\n";

    bool random = false, perturb_egs = false;
    int32 srand_seed = 0;
    int32 frame_shift = 0;
    int32 truncate_deriv_weights = 0;
    BaseFloat keep_proportion = 1.0;
    std::string spk_filter1 = "",
      input_spk_list = "";
    BaseFloat max_speed_perturb = 0.0,
      max_rand_shift = 0.0;
    ParseOptions po(usage);
    po.Register("random", &random, "If true, will write frames to output "
                "archives randomly, not round-robin.");
    po.Register("perturb_egs", &perturb_egs, "If true, different type of perturbation"
                "are applied on features stored in egs and egs are behaved as raw-waveform");
    po.Register("max-rand-shift", &max_rand_shift,"max rand shift used to randomly shift features"
                "as raw-waveform");
    po.Register("max-speed-perturb", &max_speed_perturb, "max speed perturbation applied on "
                "features stored in egs.(assumed features are raw-waveform.");
    po.Register("spk-filter1", &spk_filter1, "speaker filter list applied on speaker correspond"
                "to utterance to remove speaker level information");
    po.Register("spk-list", &input_spk_list, "list of speaker used to randomly generate a random speaker"
                "to use as a filter per utterance");
    po.Register("keep-proportion", &keep_proportion, "If <1.0, this program will "
                "randomly keep this proportion of the input samples.  If >1.0, it will "
                "in expectation copy a sample this many times.  It will copy it a number "
                "of times equal to floor(keep-proportion) or ceil(keep-proportion).");
    po.Register("srand", &srand_seed, "Seed for random number generator "
                "(only relevant if --random=true or --keep-proportion != 1.0)");
    po.Register("frame-shift", &frame_shift, "Allows you to shift time values "
                "in the supervision data (excluding iVector data) - useful in "
                "augmenting data.  Note, the outputs will remain at the closest "
                "exact multiples of the frame subsampling factor");
    po.Register("truncate-deriv-weights", &truncate_deriv_weights,
                "If nonzero, the number of initial/final subsample frames that "
                "will have their derivatives' weights set to zero.");

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }
    std::string examples_rspecifier = po.GetArg(1);

    SequentialNnetChainExampleReader example_reader(examples_rspecifier);
     
    int32 num_outputs = po.NumArgs() - 1;
    std::vector<NnetChainExampleWriter*> example_writers(num_outputs);
    for (int32 i = 0; i < num_outputs; i++)
      example_writers[i] = new NnetChainExampleWriter(po.GetArg(i+2));

    std::vector<std::string> exclude_names; // names we never shift times of;
                                            // not configurable for now.
    exclude_names.push_back(std::string("ivector"));


    int64 num_read = 0, num_written = 0;
    if (perturb_egs) {
      KALDI_ASSERT(spk_filter1 != "");
      
      RandomAccessBaseFloatVectorReader spkf1_reader(spk_filter1); 
      //RandomAccessTokenVectorReader utt2rec_reader(utt2rec);
      // read speaker lists
      std::string spk_list_str;
      {
        Input ki(input_spk_list);
        std::getline(ki.Stream(), spk_list_str);
        //spk_list_str.Read(ki.Stream());
      }
      
      std::vector<std::string> spk_list;
      SplitStringToVector(spk_list_str, " ", true, &spk_list);
      int32 num_spks = spk_list.size();

      for (; !example_reader.Done(); example_reader.Next(), num_read++) {
        // count is normally 1; could be 0, or possibly >1.
        int32 count = GetCount(keep_proportion);
        std::string key = example_reader.Key();
        if (frame_shift == 0 && truncate_deriv_weights == 0) {
          const NnetChainExample &eg = example_reader.Value();
          for (int32 c = 0; c < count; c++) {
            int32 index = (random ? Rand() : num_written) % num_outputs;
            example_writers[index]->Write(key, eg);
            num_written++;
          }
        } else if (count > 0) {
          NnetChainExample eg = example_reader.Value();
          int32 spk_num = RandInt(0, num_spks - 1);
          std::string rand_spk = spk_list[spk_num];
          // extract utt_id from eg's utt_id which is "utt_id"_"start_frame"
          std::vector<std::string> split_utt;
          SplitStringToVector(key, "_", true, &split_utt);
          const Vector<BaseFloat> *spkf1 = NULL;
          if (spkf1_reader.HasKey(split_utt[0])) {
            spkf1 = &(spkf1_reader.Value(split_utt[0]));
          } else {
            std::vector<std::string> split_spk;
            SplitStringToVector(split_utt[0], "-", true, &split_spk);
            std::string new_spk_name = split_spk[1];
            for (int32 i = 2; i < split_spk.size(); i++)
              new_spk_name = new_spk_name+"-"+split_spk[i];
              if (spkf1_reader.HasKey(new_spk_name))
                spkf1 = &(spkf1_reader.Value(new_spk_name));
              else
                KALDI_ERR << "No speaker filter for speaker-id " << new_spk_name;
          }
          const Vector<BaseFloat> *spkf2 = NULL;
          if (spkf1_reader.HasKey(rand_spk)) 
            spkf2 = &(spkf1_reader.Value(rand_spk));
          if (spkf2 == NULL) 
            KALDI_ERR << "no speaker filter for speaker " << rand_spk;

          KALDI_ASSERT(spkf1->Dim() != 0 && spkf2->Dim() != 0);
          if (perturb_egs) 
            PerturbRawChainExample(max_rand_shift, max_speed_perturb,
            (spkf1->Dim() != 0 ? spkf1 : NULL),
            (spkf2->Dim() != 0 ? spkf2 : NULL), &eg);
          
          if (frame_shift != 0)
            ShiftChainExampleTimes(frame_shift, exclude_names, &eg);

          if (truncate_deriv_weights != 0)
            TruncateDerivWeights(truncate_deriv_weights, &eg);
          for (int32 c = 0; c < count; c++) {
            int32 index = (random ? Rand() : num_written) % num_outputs;
            example_writers[index]->Write(key, eg);
            num_written++;
          }
        }
      }
    } else {
      for (; !example_reader.Done(); example_reader.Next(), num_read++) {
        // count is normally 1; could be 0, or possibly >1.
        int32 count = GetCount(keep_proportion);
        std::string key = example_reader.Key();
        if (frame_shift == 0 && truncate_deriv_weights == 0) {
          const NnetChainExample &eg = example_reader.Value();
          for (int32 c = 0; c < count; c++) {
            int32 index = (random ? Rand() : num_written) % num_outputs;
            example_writers[index]->Write(key, eg);
            num_written++;
          }
        } else if (count > 0) {
          NnetChainExample eg = example_reader.Value();
          if (frame_shift != 0)
            ShiftChainExampleTimes(frame_shift, exclude_names, &eg);
          if (truncate_deriv_weights != 0)
            TruncateDerivWeights(truncate_deriv_weights, &eg);
          for (int32 c = 0; c < count; c++) {
            int32 index = (random ? Rand() : num_written) % num_outputs;
            example_writers[index]->Write(key, eg);
            num_written++;
          }
        }
      }
    }
    for (int32 i = 0; i < num_outputs; i++)
      delete example_writers[i];
    KALDI_LOG << "Read " << num_read
              << " neural-network training examples, wrote " << num_written;
    return (num_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


