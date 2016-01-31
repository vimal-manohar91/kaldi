// nnet3bin/nnet3-discriminative-copy-egs-to-nnet2.cc

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

void AppendVectorToFeats(const MatrixBase<BaseFloat> &in,
                         const VectorBase<BaseFloat> &vec,
                         Matrix<BaseFloat> *out) {
  KALDI_ASSERT(in.NumRows() != 0);
  out->Resize(in.NumRows(), in.NumCols() + vec.Dim());
  out->Range(0, in.NumRows(),
             0, in.NumCols()).CopyFromMat(in);
  out->Range(0, in.NumRows(),
             in.NumCols(), vec.Dim()).CopyRowsFromVec(vec);
}

void ConvertToNnet2(const nnet3::NnetDiscriminativeExample &eg,
                    nnet2::DiscriminativeNnetExample *nnet2_eg) {
  nnet2_eg->num_ali = eg.outputs[0].supervision.num_ali;
  ConvertLattice(eg.outputs[0].supervision.den_lat, &nnet2_eg->den_lat);

  int32 feat_dim = eg.inputs[0].features.NumCols();
  int32 ivector_dim = eg.inputs[1].features.NumCols();
  int32 nrows = eg.inputs[0].features.NumRows();

  if (eg.inputs[1].features.NumRows() == 1) {
    Matrix<BaseFloat> feats(nrows, feat_dim);
    eg.inputs[0].features.CopyToMat(&feats, kNoTrans);
    Matrix<BaseFloat> ivector(1, ivector_dim);
    eg.inputs[1].features.CopyToMat(&ivector, kNoTrans);

    AppendVectorToFeats(feats, ivector.Row(0),
                        &(nnet2_eg->input_frames));
  } else {
    nnet2_eg->input_frames.Resize(nrows, feat_dim + ivector_dim);
    SubMatrix<BaseFloat> feats(nnet2_eg->input_frames, 0, nrows, 0, feat_dim);
    eg.inputs[0].features.CopyToMat(&feats, kNoTrans);
    SubMatrix<BaseFloat> ivector(nnet2_eg->input_frames, 0, nrows, feat_dim, ivector_dim);
    eg.inputs[1].features.CopyToMat(&ivector, kNoTrans);
  }

  nnet2_eg->left_context = -eg.inputs[0].indexes[0].t;
  nnet2_eg->weight = eg.outputs[0].supervision.weight;
  nnet2_eg->Check();
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy examples for nnet3 discriminative training, possibly changing the binary mode.\n"
        "into nnet2 examples\n"
        "\n"
        "Usage:  nnet3-discriminative-copy-egs [options] <egs-rspecifier> <egs-wspecifier1> [<egs-wspecifier2> ...]\n"
        "\n"
        "e.g.\n"
        "nnet3-discriminative-copy-egs-to-nnet2 ark:train.degs ark,t:text.degs\n"
        "or:\n"
        "nnet3-discriminative-copy-egs-to-nnet2 ark:train.degs ark:1.degs ark:2.degs\n";

    bool random = false;
    int32 srand_seed = 0;
    int32 frame_shift = 0;
    int32 truncate_deriv_weights = 0;
    BaseFloat keep_proportion = 1.0;

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

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1);

    SequentialNnetDiscriminativeExampleReader example_reader(examples_rspecifier);

    int32 num_outputs = po.NumArgs() - 1;
    std::vector<nnet2::DiscriminativeNnetExampleWriter*> example_writers(num_outputs);
    for (int32 i = 0; i < num_outputs; i++)
      example_writers[i] = new nnet2::DiscriminativeNnetExampleWriter(po.GetArg(i+2));

    std::vector<std::string> exclude_names; // names we never shift times of;
                                            // not configurable for now.
    exclude_names.push_back(std::string("ivector"));


    int64 num_read = 0, num_written = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      // count is normally 1; could be 0, or possibly >1.
      int32 count = GetCount(keep_proportion);
      std::string key = example_reader.Key();
      if (frame_shift == 0 && truncate_deriv_weights == 0) {
        const NnetDiscriminativeExample &eg = example_reader.Value();
        for (int32 c = 0; c < count; c++) {
          int32 index = (random ? Rand() : num_written) % num_outputs;
          nnet2::DiscriminativeNnetExample nnet2_eg;
          ConvertToNnet2(eg, &nnet2_eg);
          example_writers[index]->Write(key, nnet2_eg);
          num_written++;
        }
      } else if (count > 0) {
        NnetDiscriminativeExample eg = example_reader.Value();
        if (frame_shift != 0)
          ShiftDiscriminativeExampleTimes(frame_shift, exclude_names, &eg);
        if (truncate_deriv_weights != 0)
          TruncateDerivWeights(truncate_deriv_weights, &eg);
        for (int32 c = 0; c < count; c++) {
          int32 index = (random ? Rand() : num_written) % num_outputs;
          nnet2::DiscriminativeNnetExample nnet2_eg;
          ConvertToNnet2(eg, &nnet2_eg);
          example_writers[index]->Write(key, nnet2_eg);
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


