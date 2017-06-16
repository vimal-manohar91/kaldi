// nnet3bin/nnet3-get-egs-multiple-targets.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//           2014-2016  Vimal Manohar

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
#include "nnet3/nnet-example-utils.h"

namespace kaldi {
namespace nnet3 {

bool ToBool(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);

  if ((str.compare("true") == 0) || (str.compare("t") == 0)
      || (str.compare("1") == 0)) 
    return true;
  if ((str.compare("false") == 0) || (str.compare("f") == 0)
      || (str.compare("0") == 0)) 
    return false;
  KALDI_ERR << "Invalid format for boolean argument [expected true or false]: "
            << str;
  return false;  // never reached
}

static bool ProcessFile(
    const MatrixBase<BaseFloat> &feats,
    const MatrixBase<BaseFloat> *ivector_feats,
    int32 ivector_period,
    const std::vector<std::string> &output_names,
    const std::vector<int32> &output_dims,
    const std::vector<const MatrixBase<BaseFloat>* > &dense_target_matrices,
    const std::vector<const Posterior*> &posteriors,
    const std::vector<const VectorBase<BaseFloat>* > &deriv_weights,
    const std::string &utt_id,
    bool compress_input,
    const std::vector<bool> &compress_targets,
    UtteranceSplitter *utt_splitter,
    NnetExampleWriter *example_writer) {
  int32 num_input_frames = feats.NumRows();
  KALDI_ASSERT(output_names.size() > 0);  // at least one output required 
  
  std::vector<ChunkTimeInfo> chunks;

  utt_splitter->GetChunksForUtterance(num_input_frames, &chunks);

  if (chunks.empty()) {
    KALDI_WARN << "Not producing egs for utterance " << utt_id
               << " because it is too short: "
               << num_input_frames << " frames.";
    return false;
  }

  // 'frame_subsampling_factor' is not used in any recipes at the time of
  // writing, this is being supported to unify the code with the 'chain' recipes
  // and in case we need it for some reason in future.
  int32 frame_subsampling_factor =
      utt_splitter->Config().frame_subsampling_factor;

  for (size_t c = 0; c < chunks.size(); c++) {
    const ChunkTimeInfo &chunk = chunks[c];

    int32 tot_input_frames = chunk.left_context + chunk.num_frames +
        chunk.right_context;

    Matrix<BaseFloat> input_frames(tot_input_frames, feats.NumCols(),
                                   kUndefined);

    int32 start_frame = chunk.first_frame - chunk.left_context;
    for (int32 t = start_frame; t < start_frame + tot_input_frames; t++) {
      int32 t2 = t;
      if (t2 < 0) t2 = 0;
      if (t2 >= num_input_frames) t2 = num_input_frames - 1;
      int32 j = t - start_frame;
      SubVector<BaseFloat> src(feats, t2),
          dest(input_frames, j);
      dest.CopyFromVec(src);
    }

    NnetExample eg;

    // call the regular input "input".
    eg.io.push_back(NnetIo("input", -chunk.left_context, input_frames));

    if (ivector_feats != NULL) {
      // if applicable, add the iVector feature.
      // choose iVector from a random frame in the chunk
      int32 ivector_frame = RandInt(start_frame,
                                    start_frame + num_input_frames - 1),
          ivector_frame_subsampled = ivector_frame / ivector_period;
      if (ivector_frame_subsampled < 0)
        ivector_frame_subsampled = 0;
      if (ivector_frame_subsampled >= ivector_feats->NumRows())
        ivector_frame_subsampled = ivector_feats->NumRows() - 1;
      Matrix<BaseFloat> ivector(1, ivector_feats->NumCols());
      ivector.Row(0).CopyFromVec(ivector_feats->Row(ivector_frame_subsampled));
      eg.io.push_back(NnetIo("ivector", 0, ivector));
    }

    // Note: chunk.first_frame and chunk.num_frames will both be
    // multiples of frame_subsampling_factor.
    // We expect frame_subsampling_factor to usually be 1 for now.
    int32 start_frame_subsampled = chunk.first_frame / frame_subsampling_factor,
        num_frames_subsampled = chunk.num_frames / frame_subsampling_factor;

    int32 num_outputs_added = 0;

    for (int32 n = 0; n < output_names.size(); n++) {
      int32 num_target_frames = 0;
      
      if (dense_target_matrices[n])
        num_target_frames = dense_target_matrices[n]->NumRows();
      else if (posteriors[n]) 
        num_target_frames = posteriors[n]->size();
      else {
        KALDI_WARN << "No target found for output " << output_names[n];
        continue;
      }

      Vector<BaseFloat> this_deriv_weights(0);
      if (deriv_weights[n]) {
        this_deriv_weights.Resize(num_frames_subsampled);
        for (int32 i = 0; i < num_frames_subsampled; i++) {
          int32 t = i + start_frame_subsampled;
          if (t >= num_target_frames) t = num_target_frames - 1;
          this_deriv_weights(i) = (*(deriv_weights[n]))(t);
        }
      }

      if (dense_target_matrices[n]) {
        const MatrixBase<BaseFloat> &targets = *dense_target_matrices[n];
        KALDI_ASSERT(start_frame_subsampled + num_frames_subsampled - 1 <
                     targets.NumRows());
        Matrix<BaseFloat> targets_part(num_frames_subsampled, 
                                       targets.NumCols());
        for (int32 i = 0; i < num_frames_subsampled; i++) {
          // Copy the i^th row of the target matrix from the (t+i)^th row of the
          // input targets matrix
          int32 t = i + start_frame_subsampled;
          if (t >= num_target_frames) t = num_target_frames - 1;
          SubVector<BaseFloat> this_target_dest(targets_part, i);
          SubVector<BaseFloat> this_target_src(targets, t);
          this_target_dest.CopyFromVec(this_target_src);
        }

        if (deriv_weights[n]) {
          eg.io.push_back(NnetIo(output_names[n], this_deriv_weights, 
                                 0, targets_part));
        } else {
          eg.io.push_back(NnetIo(output_names[n], 0, targets_part));
        }
      } else {
        const Posterior &pdf_post = *(posteriors[n]);
        //KALDI_ASSERT(start_frame_subsampled + num_frames_subsampled - 1 <
        //             pdf_post.size());
        Posterior labels(num_frames_subsampled);
        for (int32 i = 0; i < num_frames_subsampled; i++) {
          int32 t = i + start_frame_subsampled;
          if (t >= num_target_frames) t = num_target_frames - 1;
          labels[i] = pdf_post[t];
          for (std::vector<std::pair<int32, BaseFloat> >::iterator
                iter = labels[i].begin(); iter != labels[i].end(); ++iter)
            iter->second *= chunk.output_weights[i];
        }
        
        if (deriv_weights[n]) {
          eg.io.push_back(NnetIo(output_names[n], this_deriv_weights, 
                                 output_dims[n], 0, labels));
        } else {
          eg.io.push_back(NnetIo(output_names[n], output_dims[n], 0, labels));
        }
      } 
      
      eg.Compress();

      num_outputs_added++;
    }

    if (num_outputs_added != output_names.size()) continue;
      
    std::ostringstream os;
    os << utt_id << "-" << chunk.first_frame;

    std::string key = os.str(); // key is <utt_id>-<frame_id>

    KALDI_ASSERT(NumOutputs(eg) == num_outputs_added);

    example_writer->Write(key, eg);
  }
  return true;
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
        "This program is similar to nnet3-get-egs, but the targets here are "
        "dense matrices instead of posteriors (sparse matrices).\n"
        "This is useful when you want the targets to be continuous real-valued "
        "with the neural network possibly trained with a quadratic objective\n"
        "\n"
        "Usage:  nnet3-get-egs-multiple-targets [options] "
        "<features-rspecifier> <output1-name>:<targets-rspecifier1>:<num-targets1>[:<deriv-weights-rspecifier1>] "
        "[ <output2-name>:<targets-rspecifier2>:<num-targets2> ... <targets-rspecifierN> ] <egs-out>\n"
        "\n"
        "Here <outputN-name> is any random string for output node name, \n"
        "<targets_rspecifierN> is the rspecifier for either dense targets in matrix format or sparse targets in posterior format,\n"
        "and <num-targetsN> is the target dimension of output node for sparse targets or -1 for dense targets\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "nnet-get-egs-multiple-targets --left-context=12 \\\n"
        "--right-context=9 --num-frames=8 \"$feats\" \\\n"
        "output-snr:\"ark:copy-matrix ark:exp/snrs/snr.1.ark ark:- |\":-1 \n"
        "   ark:- \n";
        

    bool compress_input = true;
    int32 input_compress_format = 0; 
    int32 length_tolerance = 2;
    int32 online_ivector_period = 1;

    ExampleGenerationConfig eg_config;  // controls num-frames,
                                        // left/right-context, etc.

    std::string online_ivector_rspecifier, 
                targets_compress_formats_str,
                compress_targets_str;
    std::string output_dims_str;
    std::string output_names_str;

    ParseOptions po(usage);
    po.Register("compress-input", &compress_input, "If true, write egs in "
                "compressed format.");
    po.Register("input-compress-format", &input_compress_format, "Format for "
                "compressing input feats e.g. Use 2 for compressing wave [deprecated]");
    po.Register("compress-targets", &compress_targets_str, "CSL of whether "
                "targets must be compressed for each of the outputs");
    po.Register("targets-compress-formats", &targets_compress_formats_str,
                "Format for compressing all feats in general [deprecated]");
    po.Register("ivectors", &online_ivector_rspecifier, "Alias for "
                "--online-ivectors option, for back compatibility");
    po.Register("online-ivectors", &online_ivector_rspecifier, "Rspecifier of "
                "ivector features, as a matrix.");
    po.Register("online-ivector-period", &online_ivector_period, "Number of "
                "frames between iVectors in matrices supplied to the "
                "--online-ivectors option");
    po.Register("length-tolerance", &length_tolerance, "Tolerance for "
                "difference in num-frames between feat and ivector matrices");
    po.Register("output-dims", &output_dims_str, "CSL of output node dims");
    po.Register("output-names", &output_names_str, "CSL of output node names");
    
    eg_config.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    eg_config.ComputeDerived();
    UtteranceSplitter utt_splitter(eg_config);

    std::string feature_rspecifier = po.GetArg(1),
               examples_wspecifier = po.GetArg(po.NumArgs());

    // Read in all the training files.
    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    RandomAccessBaseFloatMatrixReader online_ivector_reader(online_ivector_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);

    int32 num_err = 0;

    int32 num_outputs = (po.NumArgs() - 2) / 2;
    KALDI_ASSERT(num_outputs > 0);
    
    std::vector<RandomAccessBaseFloatVectorReader*> deriv_weights_readers(
        num_outputs, static_cast<RandomAccessBaseFloatVectorReader*>(NULL));
    std::vector<RandomAccessBaseFloatMatrixReader*> dense_targets_readers(
        num_outputs, static_cast<RandomAccessBaseFloatMatrixReader*>(NULL));
    std::vector<RandomAccessPosteriorReader*> sparse_targets_readers(
        num_outputs, static_cast<RandomAccessPosteriorReader*>(NULL));

    std::vector<bool> compress_targets(1, true);
    std::vector<std::string> compress_targets_vector;

    if (!compress_targets_str.empty()) {
      SplitStringToVector(compress_targets_str, ":,",
                          true, &compress_targets_vector);
    }

    if (compress_targets_vector.size() == 1 && num_outputs != 1) {
      KALDI_WARN << "compress-targets is of size 1. "
                 << "Extending it to size num-outputs=" << num_outputs;
      compress_targets[0] = ToBool(compress_targets_vector[0]);
      compress_targets.resize(num_outputs, ToBool(compress_targets_vector[0]));
    } else {
      if (compress_targets_vector.size() != num_outputs) {
        KALDI_ERR << "Mismatch in length of compress-targets and num-outputs; "
                  << compress_targets_vector.size() << " vs " << num_outputs;
      }
      for (int32 n = 0; n < num_outputs; n++) {
        compress_targets[n] = ToBool(compress_targets_vector[n]);
      }
    }

    std::vector<int32> output_dims(num_outputs);
    SplitStringToIntegers(output_dims_str, ":,", 
                            true, &output_dims);

    std::vector<std::string> output_names(num_outputs);
    SplitStringToVector(output_names_str, ":,", true, &output_names);
    
    std::vector<std::string> targets_rspecifiers(num_outputs);
    std::vector<std::string> deriv_weights_rspecifiers(num_outputs);
    
    for (int32 n = 0; n < num_outputs; n++) {
      const std::string &targets_rspecifier = po.GetArg(2*n + 2);
      const std::string &deriv_weights_rspecifier = po.GetArg(2*n + 3);
  
      targets_rspecifiers[n] = targets_rspecifier;
      deriv_weights_rspecifiers[n] = deriv_weights_rspecifier;

      if (output_dims[n] >= 0) {
        sparse_targets_readers[n] = new RandomAccessPosteriorReader(
            targets_rspecifier);
      } else {
        dense_targets_readers[n] = new RandomAccessBaseFloatMatrixReader(
            targets_rspecifier);
      }

      if (!deriv_weights_rspecifier.empty())
        deriv_weights_readers[n] = new RandomAccessBaseFloatVectorReader(
            deriv_weights_rspecifier);

      KALDI_LOG << "output-name=" << output_names[n]
                << " target-dim=" << output_dims[n]
                << " targets-rspecifier=\"" << targets_rspecifiers[n] << "\""
                << " deriv-weights-rspecifier=\"" 
                << deriv_weights_rspecifiers[n] << "\""
                << " compress-target=" 
                << (compress_targets[n] ? "true" : "false");
    }

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
        
      const Matrix<BaseFloat> *online_ivector_feats = NULL;
      if (!online_ivector_rspecifier.empty()) {
        if (!online_ivector_reader.HasKey(key)) {
          KALDI_WARN << "No iVectors for utterance " << key;
          num_err++;
          continue;
        } else {
          // this address will be valid until we call HasKey() or Value()
          // again.
          online_ivector_feats = &(online_ivector_reader.Value(key));
        }
      }

      if (online_ivector_feats && 
          (abs(feats.NumRows() - online_ivector_feats->NumRows()) > length_tolerance
           || online_ivector_feats->NumRows() == 0)) {
        KALDI_WARN << "Length difference between feats " << feats.NumRows()
                   << " and iVectors " << online_ivector_feats->NumRows()
                   << "exceeds tolerance " << length_tolerance;
        num_err++;
        continue;
      }

      std::vector<const MatrixBase<BaseFloat>* > dense_targets(
          num_outputs, static_cast<const Matrix<BaseFloat>* >(NULL));
      std::vector<const Posterior* > sparse_targets(
          num_outputs, static_cast<const Posterior* >(NULL));
      std::vector<const VectorBase<BaseFloat>* > deriv_weights(
          num_outputs, static_cast<const Vector<BaseFloat>* >(NULL));

      int32 num_outputs_found = 0;
      for (int32 n = 0; n < num_outputs; n++) {
        if (dense_targets_readers[n]) {
          if (!dense_targets_readers[n]->HasKey(key)) {
            KALDI_WARN << "No dense targets matrix for key " << key << " in " 
                       << "rspecifier " << targets_rspecifiers[n] 
                       << " for output " << output_names[n];
            break;
          } 
          const MatrixBase<BaseFloat> *target_matrix = &(
              dense_targets_readers[n]->Value(key));
          
          if ((target_matrix->NumRows() - feats.NumRows()) > length_tolerance) {
            KALDI_WARN << "Length difference between feats " << feats.NumRows()
                       << " and target matrix " << target_matrix->NumRows()
                       << "exceeds tolerance " << length_tolerance;
            break;
          }

          dense_targets[n] = target_matrix;
        } else {
          if (!sparse_targets_readers[n]->HasKey(key)) {
            KALDI_WARN << "No sparse target matrix for key " << key << " in " 
                       << "rspecifier " << targets_rspecifiers[n]
                       << " for output " << output_names[n];
            break;
          } 
          const Posterior *posterior = &(sparse_targets_readers[n]->Value(key));

          if (abs(static_cast<int32>(posterior->size()) - feats.NumRows()) 
              > length_tolerance) {
            KALDI_WARN << "For key " << key 
                       << " posterior has wrong size " << posterior->size()
                       << " versus " << feats.NumRows();
            break;
          }
        
          sparse_targets[n] = posterior;
        }
        
        if (deriv_weights_readers[n]) {
          if (!deriv_weights_readers[n]->HasKey(key)) {
            KALDI_WARN << "No deriv weights for key " << key << " in " 
                       << "rspecifier " << deriv_weights_rspecifiers[n]
                       << " for output " << output_names[n];
            break;
          } else {
            // this address will be valid until we call HasKey() or Value()
            // again.
            deriv_weights[n] = &(deriv_weights_readers[n]->Value(key));
          }
        }
        
        if (deriv_weights[n] 
            && (abs(feats.NumRows() - deriv_weights[n]->Dim())
                > length_tolerance
                || deriv_weights[n]->Dim() == 0)) {
          KALDI_WARN << "Length difference between feats " << feats.NumRows()
                     << " and deriv weights " << deriv_weights[n]->Dim()
                     << " exceeds tolerance " << length_tolerance;
          break;
        }
        
        num_outputs_found++;
      }

      if (num_outputs_found != num_outputs) {
        KALDI_WARN << "Not all outputs found for key " << key;
        num_err++;
        continue;
      }

      if (!ProcessFile(feats, online_ivector_feats, online_ivector_period,
                       output_names, output_dims,
                       dense_targets, sparse_targets,
                       deriv_weights, key,
                       compress_input, compress_targets, 
                       &utt_splitter, &example_writer))
        num_err++;
    }
    for (int32 n = 0; n < num_outputs; n++) {
      delete dense_targets_readers[n];
      delete sparse_targets_readers[n];
      delete deriv_weights_readers[n];
    }
    if (num_err > 0) 
      KALDI_WARN << num_err << " utterannces had errors and could "
              "not be processed.";
    // utt_splitter prints stats in its destructor.
    return utt_splitter.ExitStatus();
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
