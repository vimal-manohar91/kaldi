// nnet3bin/nnet3-get-raw-egs.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//                2014  Vimal Manohar
//                2015  Pegah Ghahremani

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
                        const MatrixBase<BaseFloat> &targets,
                        const Posterior &pdf_post,
                        const std::string &utt_id,
                        bool compress,
                        int32 num_targets,
                        int32 num_pdfs,
                        int32 left_context,
                        int32 right_context,
                        int32 frames_per_eg,
                        int64 *num_frames_written,
                        int64 *num_egs_written,
                        NnetExampleWriter *example_writer) {
  //KALDI_ASSERT(feats.NumRows() == static_cast<int32>(pdf_post.size()));
  int32 min_size = std::min(feats.NumRows(), std::min(targets.NumRows(), static_cast<int32>(pdf_post.size())));
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

    // if applicable, add the iVector feature.
    if (ivector_feats != NULL) {
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
    // add the labels for output1
    Matrix<BaseFloat> targets_dest(frames_per_eg, targets.NumCols()); 
    for (int32 i = 0; i < actual_frames_per_eg; i++) {
      // Copy the i^th row of the target matrix from the (t+i)^th row of the 
      // input targets matrix  
      SubVector<BaseFloat> this_target_dest(targets_dest, i);
      SubVector<BaseFloat> this_target_src(targets, t+i);
      this_target_dest.CopyFromVec(this_target_src);
    }

    // Copy the last frame's target to the padded frames
    for (int32 i = actual_frames_per_eg; i < frames_per_eg; i++) {
      // Copy the i^th row of the target matrix from the last row of the 
      // input targets matrix
      KALDI_ASSERT(t + actual_frames_per_eg - 1 == min_size - 1);
      SubVector<BaseFloat> this_target_dest(targets_dest, i);
      SubVector<BaseFloat> this_target_src(targets, t+actual_frames_per_eg-1);
      this_target_dest.CopyFromVec(this_target_src);
    } 

    // push this created targets matrix into the eg   
    eg.io.push_back(NnetIo("output2", 0, targets_dest));

    // add the labels for output2.
    Posterior labels(frames_per_eg);
    for (int32 i = 0; i < actual_frames_per_eg; i++)
      labels[i] = pdf_post[t + i];
    // remaining posteriors for frames are empty.
    eg.io.push_back(NnetIo("output", num_pdfs, 0, labels));
    
    if (compress)
      eg.Compress(2.0);
      
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
        "This program is similar to nnet3-get-egs, but the examples can have "
        " 2 outputs, where the 1st output is a dense matrix and the second output "
        " is a sparse matrix (posterior).\n"
        "This is useful when you want to have two sets of continuous real-valued output "
        " with a quadratic objective and discrete output with CE objective. \n"
        "\n"
        "Usage:  nnet3-get-egs [options] <features-rspecifier> "
        "<target-rspecifier> <pdf-post-rspecifier> <egs-out>\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "nnet3-get-egs --num-pdfs=2658 --left-context=12 --right-context=9 --num-frames=8 \"$feats\"\\\n"
        "\"ark:copy-feats scp:data/split4/1/feats.scp ark:-|\"\n"
        "\"ark:gunzip -c exp/nnet/ali.1.gz | ali-to-pdf exp/nnet/1.nnet ark:- ark:- | ali-to-post ark:- ark:- |\" \\\n"
        "   ark:- \n";
        

    bool compress = true;
    int32 num_pdfs = -1, num_targets = -1, left_context = 0, right_context = 0,
        num_frames = 1, length_tolerance = 100;
        
    std::string ivector_rspecifier;
    
    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs in "
                "compressed format.");
    po.Register("num-pdfs", &num_pdfs, "Number of pdfs correspond to 2nd label in the acoustic "
                "model");
    po.Register("num-targets", &num_targets, "Number of targets correspond to 1st label in"
                "the neural network");
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
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    if (num_pdfs <= 0)
      KALDI_ERR << "--num-pdfs options is required.";
    if (num_targets <= 0)
      KALDI_ERR << "--num-targets option is required.";

    std::string feature_rspecifier = po.GetArg(1),
        matrix_rspecifier = po.GetArg(2),
        pdf_post_rspecifier = po.GetArg(3),
        examples_wspecifier = po.GetArg(4);

    // Read in all the training files.
    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    RandomAccessBaseFloatMatrixReader matrix_reader(matrix_rspecifier);
    RandomAccessPosteriorReader pdf_post_reader(pdf_post_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);
    RandomAccessBaseFloatMatrixReader ivector_reader(ivector_rspecifier);
    
    int32 num_done = 0, num_err = 0;
    int64 num_frames_written = 0, num_egs_written = 0;
    
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      if (!matrix_reader.HasKey(key)) {
        KALDI_WARN << "No target matrix for key " << key;  
        num_err++;
      } else {
        const Matrix<BaseFloat> &target_matrix = matrix_reader.Value(key);
        KALDI_ASSERT(target_matrix.NumCols() == num_targets);
        if ((target_matrix.NumRows() - feats.NumRows()) > length_tolerance) {
          KALDI_WARN << "Length difference between feats " << feats.NumRows()
                     << " and target matrix " << target_matrix.NumRows()
                     << "exceeds tolerance " << length_tolerance;
          num_err++;
          continue;
        }
        if (!pdf_post_reader.HasKey(key)) {
          KALDI_WARN << "No pdf-level posterior for key " << key;
          num_err++;
        } else {
          const Posterior &pdf_post = pdf_post_reader.Value(key);
          if (abs(pdf_post.size() - feats.NumRows()) > length_tolerance) {
            KALDI_WARN << "Length difference between feats " << feats.NumRows()
                       << " and posteriors " << pdf_post.size() 
                       << "exceeds tolerance " << length_tolerance;
            num_err++;
            continue;
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

          if (ivector_feats != NULL &&
              (abs(feats.NumRows() - ivector_feats->NumRows()) > length_tolerance
               || ivector_feats->NumRows() == 0)) {
            KALDI_WARN << "Length difference between feats " << feats.NumRows()
                       << " and iVectors " << ivector_feats->NumRows()
                       << "exceeds tolerance " << length_tolerance;
            num_err++;
            continue;
          }
          ProcessFile(feats, ivector_feats, target_matrix, pdf_post, key, compress, 
                      num_targets, num_pdfs, left_context, right_context, num_frames,
                      &num_frames_written, &num_egs_written,
                      &example_writer);
          num_done++;
        }
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
