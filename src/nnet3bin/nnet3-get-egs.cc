// nnet3bin/nnet3-get-egs.cc

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

#include <sstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include "nnet3/nnet-example.h"

namespace kaldi {
namespace nnet3 {


static void ProcessFile(const MatrixBase<BaseFloat> &feats,
                        const MatrixBase<BaseFloat> *ivector_feats,
                        const VectorBase<BaseFloat> *deriv_weights,
                        const Posterior &pdf_post,
                        const std::string &utt_id,
                        bool compress,
                        int32 input_compress_format,
                        int32 feats_compress_format,
                        int32 num_pdfs,
                        int32 left_context,
                        int32 right_context,
                        int32 frames_per_eg,
                        int64 *num_frames_written,
                        int64 *num_egs_written,
                        NnetExampleWriter *example_writer) {
  //KALDI_ASSERT(feats.NumRows() == static_cast<int32>(pdf_post.size()));
  int32 min_size = std::min(feats.NumRows(), static_cast<int32>(pdf_post.size()));
  for (int32 t = 0; t < min_size; t += frames_per_eg) {

    // actual_frames_per_eg is the number of frames with nonzero
    // posteriors.  At the end of the file we pad with zero posteriors
    // so that all examples have the same structure (prevents the need
    // for recompilations).
    int32 actual_frames_per_eg = std::min(frames_per_eg,
                                          min_size - t);


    int32 tot_frames = left_context + frames_per_eg + right_context;

    Matrix<BaseFloat> input_frames(tot_frames, feats.NumCols(), kUndefined);
    
    // Set up "input_frames".
    for (int32 j = -left_context; j < frames_per_eg + right_context; j++) {
      int32 t2 = j + t;
      if (t2 < 0) t2 = 0;
      if (t2 >= min_size) t2 = min_size - 1;
      SubVector<BaseFloat> src(feats, t2),
          dest(input_frames, j + left_context);
      dest.CopyFromVec(src);
    }

    NnetExample eg;
    
    // call the regular input "input".
    eg.io.push_back(NnetIo("input", - left_context,
                           input_frames));
    
    if (compress)
      eg.io.back().Compress(input_compress_format);

    // if applicable, add the iVector feature.
    if (ivector_feats) {
      // try to get closest frame to middle of window to get
      // a representative iVector.
      int32 closest_frame = t + (actual_frames_per_eg / 2);
      KALDI_ASSERT(ivector_feats->NumRows() > 0);
      if (closest_frame >= ivector_feats->NumRows())
        closest_frame = ivector_feats->NumRows() - 1;
      Matrix<BaseFloat> ivector(1, ivector_feats->NumCols());
      ivector.Row(0).CopyFromVec(ivector_feats->Row(closest_frame));
      eg.io.push_back(NnetIo("ivector", 0, ivector));
    }

    // add the labels.
    Posterior labels(frames_per_eg);
    for (int32 i = 0; i < actual_frames_per_eg; i++)
      labels[i] = pdf_post[t + i];
    // remaining posteriors for frames are empty.

    if (!deriv_weights) {
      eg.io.push_back(NnetIo("output", num_pdfs, 0, labels));
    } else {
      Vector<BaseFloat> this_deriv_weights(frames_per_eg);
      int32 frames_to_copy = std::min(t + actual_frames_per_eg, deriv_weights->Dim()) - t; 
      this_deriv_weights.Range(0, frames_to_copy).CopyFromVec(deriv_weights->Range(t, frames_to_copy));
      if (this_deriv_weights.Sum() == 0) continue;  // Ignore frames that have frame weights 0
      eg.io.push_back(NnetIo("output", this_deriv_weights, num_pdfs, 0, labels));
    }

    
    if (compress)
      eg.Compress(feats_compress_format);
      
    std::ostringstream os;
    os << utt_id << "-" << t;

    std::string key = os.str(); // key is <utt_id>-<frame_id>

    *num_frames_written += actual_frames_per_eg;
    *num_egs_written += 1;

    example_writer->Write(key, eg);
  }
}


} // namespace nnet2
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get frame-by-frame examples of data for nnet3 neural network training.\n"
        "Essentially this is a format change from features and posteriors\n"
        "into a special frame-by-frame format.  This program handles the\n"
        "common case where you have some input features, possibly some\n"
        "iVectors, and one set of labels.  If people in future want to\n"
        "do different things they may have to extend this program or create\n"
        "different versions of it for different tasks (the egs format is quite\n"
        "general)\n"
        "\n"
        "Usage:  nnet3-get-egs [options] <features-rspecifier> "
        "<pdf-post-rspecifier> <egs-out>\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "nnet3-get-egs --num-pdfs=2658 --left-context=12 --right-context=9 --num-frames=8 \"$feats\"\\\n"
        "\"ark:gunzip -c exp/nnet/ali.1.gz | ali-to-pdf exp/nnet/1.nnet ark:- ark:- | ali-to-post ark:- ark:- |\" \\\n"
        "   ark:- \n";
        

    bool compress = true;
    int32 input_compress_format = 0, feats_compress_format = 0;
    int32 num_pdfs = -1, left_context = 0, right_context = 0,
        num_frames = 1, length_tolerance = 100;
        
    std::string ivector_rspecifier, deriv_weights_rspecifier;
    
    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs in "
                "compressed format.");
    po.Register("compress-format", &feats_compress_format, "Format for "
                "compressing all feats in general");
    po.Register("input-compress-format", &input_compress_format, "Format for "
                "compressing input feats e.g. Use 2 for compressing wave");
    po.Register("num-pdfs", &num_pdfs, "Number of pdfs in the acoustic "
                "model");
    po.Register("left-context", &left_context, "Number of frames of left "
                "context the neural net requires.");
    po.Register("right-context", &right_context, "Number of frames of right "
                "context the neural net requires.");
    po.Register("num-frames", &num_frames, "Number of frames with labels "
                "that each example contains.");
    po.Register("ivectors", &ivector_rspecifier, "Rspecifier of ivector "
                "features, as a matrix.");
    po.Register("length-tolerance", &length_tolerance, "Tolerance for "
                "difference in num-frames between feat and ivector matrices");
    po.Register("deriv-weights-rspecifier", &deriv_weights_rspecifier,
                "Per-frame weights (only binary - 0 or 1) that specifies "
                "whether a frame's gradient must be backpropagated or not. "
                "Not specifying this is equivalent to specifying a vector of "
                "all 1s.");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    if (num_pdfs <= 0)
      KALDI_ERR << "--num-pdfs options is required.";
    

    std::string feature_rspecifier = po.GetArg(1),
        pdf_post_rspecifier = po.GetArg(2),
        examples_wspecifier = po.GetArg(3);

    // Read in all the training files.
    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    RandomAccessPosteriorReader pdf_post_reader(pdf_post_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);
    RandomAccessBaseFloatMatrixReader ivector_reader(ivector_rspecifier);
    RandomAccessBaseFloatVectorReader deriv_weights_reader(deriv_weights_rspecifier);
    
    int32 num_done = 0, num_err = 0;
    int64 num_frames_written = 0, num_egs_written = 0;
    
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      if (!pdf_post_reader.HasKey(key)) {
        KALDI_WARN << "No pdf-level posterior for key " << key;
        num_err++;
      } else {
        Posterior pdf_post = pdf_post_reader.Value(key);
        if (abs(static_cast<int32>(pdf_post.size()) - feats.NumRows()) > length_tolerance
            || pdf_post.size() < feats.NumRows()) {
          KALDI_WARN << "Posterior has wrong size " << pdf_post.size()
                     << " versus " << feats.NumRows();
          num_err++;
          continue;
        }
        while (static_cast<int32>(pdf_post.size()) > feats.NumRows()) {
          pdf_post.pop_back();
        }
        const Matrix<BaseFloat> *ivector_feats = NULL;
        if (!ivector_rspecifier.empty()) {
          if (!ivector_reader.HasKey(key)) {
            KALDI_WARN << "No iVectors for utterance " << key;
            num_err++;
            continue;
          } else {
            // this address will be valid until we call HasKey() or Value()
            // again.
            ivector_feats = &(ivector_reader.Value(key));
          }
        }

        if (ivector_feats &&
            (abs(feats.NumRows() - ivector_feats->NumRows()) > length_tolerance
             || ivector_feats->NumRows() == 0)) {
          KALDI_WARN << "Length difference between feats " << feats.NumRows()
                     << " and iVectors " << ivector_feats->NumRows()
                     << "exceeds tolerance " << length_tolerance;
          num_err++;
          continue;
        }

        const Vector<BaseFloat> *deriv_weights = NULL;
        if (!deriv_weights_rspecifier.empty()) {
          if (!deriv_weights_reader.HasKey(key)) {
            KALDI_WARN << "No deriv weights for utterance " << key;
            num_err++;
            continue;
          } else {
            // this address will be valid until we call HasKey() or Value()
            // again.
            deriv_weights = &(deriv_weights_reader.Value(key));
          }
        }

        if (deriv_weights && 
            (abs(feats.NumRows() - deriv_weights->Dim()) > length_tolerance
            || deriv_weights->Dim() == 0)) {
          KALDI_WARN << "Length difference between feats " << feats.NumRows()
                     << " and deriv weights " << deriv_weights->Dim()
                     << " exceeds tolerance " << length_tolerance;
          num_err++;
          continue;
        }

          
        ProcessFile(feats, ivector_feats, deriv_weights, pdf_post, 
                    key, compress, input_compress_format, feats_compress_format, 
                    num_pdfs, left_context, right_context, num_frames,
                    &num_frames_written, &num_egs_written,
                    &example_writer);
        num_done++;
      }
    }

    KALDI_LOG << "Finished generating examples, "
              << "successfully processed " << num_done
              << " feature files, wrote " << num_egs_written << " examples, "
              << " with " << num_frames_written << " egs in total; "
              << num_err << " files had errors.";
    return (num_egs_written == 0 || num_err > num_done ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
