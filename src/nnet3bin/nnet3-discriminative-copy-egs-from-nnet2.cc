// nnet3bin/nnet3-discriminative-copy-egs-from-nnet2.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//           2014-2015  Vimal Manohar

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
#include "nnet3/nnet-discriminative-example.h"
#include "nnet2/nnet-example.h"
#include "nnet2/nnet-example-functions.h"

namespace kaldi {
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

void ConvertToNnet3(const nnet2::DiscriminativeNnetExample &eg,
                    int32 fixed_vector_dim,
                    nnet3::NnetDiscriminativeExample *nnet3_eg) {
  nnet3_eg->outputs.resize(1);
  nnet3_eg->outputs[0].supervision.num_ali = eg.num_ali;
  ConvertLattice(eg.den_lat, &(nnet3_eg->outputs[0].supervision.den_lat));
  fst::TopSort(&(nnet3_eg->outputs[0].supervision.den_lat));

  KALDI_ASSERT(eg.spk_info.Dim() == 0 || fixed_vector_dim >= 0);
  // If eg.spk_info has zero dimension, the fixed vector might be in the 
  // input_frames. Then fixed_vector_dim will be used to determine the 
  // dimension of the speaker info
  int32 ivector_dim = eg.spk_info.Dim();
  int32 feat_dim = eg.input_frames.NumCols();

  if (eg.spk_info.Dim() == 0) {
    ivector_dim = fixed_vector_dim;
    feat_dim -= fixed_vector_dim;
  }
  KALDI_ASSERT(feat_dim > 0);
  int32 nrows = eg.input_frames.NumRows();

  nnet3_eg->inputs.resize(ivector_dim > 0 ? 2 : 1);

  nnet3_eg->inputs[0].name = "input";
  SubMatrix<BaseFloat> feats(eg.input_frames, 0, nrows, 0, feat_dim);
  nnet3_eg->inputs[0].features = feats;

  nnet3_eg->inputs[0].indexes.resize(nrows);
  for (int32 i = 0; i < nrows; i++) {
    nnet3_eg->inputs[0].indexes[i] = nnet3::Index(0, i - eg.left_context, 0);
  }
  
  if (ivector_dim > 0) {
    nnet3_eg->inputs[1].name = "ivector";
    Matrix<BaseFloat> ivector;
    
    if (eg.spk_info.Dim() == 0) {
      ivector.Resize(nrows, ivector_dim);
      ivector.CopyFromMat(eg.input_frames.Range(0, nrows, feat_dim, ivector_dim));
      nnet3_eg->inputs[1].indexes.resize(nrows);
      for (int32 i = 0; i < nrows; i++) {
        nnet3_eg->inputs[1].indexes[i] = nnet3::Index(0, i - eg.left_context, 0);
      }
    } else {
      ivector.Resize(1, ivector_dim);
      ivector.Row(0).CopyFromVec(eg.spk_info);
      nnet3_eg->inputs[1].indexes.resize(1);
      nnet3_eg->inputs[1].indexes[0] = nnet3::Index(0, 0, 0);
    }
    nnet3_eg->inputs[1].features = ivector;
  }

  nnet3_eg->outputs[0].name = "output";
  nnet3_eg->outputs[0].indexes.resize(eg.num_ali.size());
  for (int32 i = 0; i < eg.num_ali.size(); i++) {
    nnet3_eg->outputs[0].indexes[i] = nnet3::Index(0, i, 0);
  }
  
  nnet3_eg->outputs[0].supervision.weight = eg.weight;
  nnet3_eg->outputs[0].supervision.frames_per_sequence = eg.num_ali.size();
  nnet3_eg->outputs[0].supervision.num_sequences = 1;
  nnet3_eg->outputs[0].CheckDim();
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy examples to nnet3 discriminative training, possibly changing the binary mode \n"
        "from nnet2 examples\n"
        "\n"
        "Usage:  nnet3-discriminative-copy-egs-from-nnet2 [options] <egs-rspecifier> <egs-wspecifier1> [<egs-wspecifier2> ...]\n"
        "\n"
        "e.g.\n"
        "nnet3-discriminative-copy-egs-from-nnet2 ark:train.degs ark,t:text.degs\n"
        "or:\n"
        "nnet3-discriminative-copy-egs-from-nnet2 ark:train.degs ark:1.degs ark:2.degs\n";

    bool random = false;
    int32 srand_seed = 0;
    int32 frame_shift = 0;
    int32 truncate_deriv_weights = 0;
    BaseFloat keep_proportion = 1.0;
    int32 max_length = -1;
    int32 fixed_vector_dim = -1;

    ParseOptions po(usage);
    po.Register("random", &random, "If true, will write frames to output "
                "archives randomly, not round-robin.");
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
    po.Register("max-length", &max_length, "Pads small segments so that "
                "they are all of the size max_length");
    po.Register("fixed-vector-dim", &fixed_vector_dim, "i-vector dimension");

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string examples_rspecifier = po.GetArg(1);

    nnet2::SequentialDiscriminativeNnetExampleReader example_reader(examples_rspecifier);

    int32 num_outputs = po.NumArgs() - 1;
    std::vector<NnetDiscriminativeExampleWriter*> example_writers(num_outputs);
    for (int32 i = 0; i < num_outputs; i++)
      example_writers[i] = new NnetDiscriminativeExampleWriter(po.GetArg(i+2));

    std::vector<std::string> exclude_names; // names we never shift times of;
                                            // not configurable for now.
    exclude_names.push_back(std::string("ivector"));


    int64 num_read = 0, num_written = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      // count is normally 1; could be 0, or possibly >1.
      int32 count = GetCount(keep_proportion);
      std::string key = example_reader.Key();
      nnet2::DiscriminativeNnetExample eg = example_reader.Value();
      if (max_length >= 0 && !PadDiscriminativeExamples(max_length, &eg)) continue;

      NnetDiscriminativeExample nnet3_eg;
      ConvertToNnet3(eg, fixed_vector_dim, &nnet3_eg);
      if (frame_shift == 0 && truncate_deriv_weights == 0) {
        for (int32 c = 0; c < count; c++) {
          int32 index = (random ? Rand() : num_written) % num_outputs;
          example_writers[index]->Write(key, nnet3_eg);
          num_written++;
        }
      } else if (count > 0) {
        if (frame_shift != 0)
          ShiftDiscriminativeExampleTimes(frame_shift, exclude_names, &nnet3_eg);
        if (truncate_deriv_weights != 0)
          TruncateDerivWeights(truncate_deriv_weights, &nnet3_eg);
        for (int32 c = 0; c < count; c++) {
          int32 index = (random ? Rand() : num_written) % num_outputs;
          example_writers[index]->Write(key, nnet3_eg);
          num_written++;
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



