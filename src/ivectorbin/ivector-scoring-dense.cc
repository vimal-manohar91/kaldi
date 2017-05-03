// ivectorbin/ivector-scoring-dense.cc

// Copyright 2016  David Snyder
//           2017  Vimal Manohar

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
#include "ivector/transform.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Perform cosine scoring for speaker diarization.  The input reco2utt\n"
      "should be of the form <recording-id> <seg1> <seg2> ... <segN> and\n"
      "there should be one iVector for each segment.  Cosine scoring is\n"
      "performed between all pairs of iVectors in a recording and outputs\n"
      "an archive of score matrices, one for each recording-id.  The rows\n"
      "and columns of the the matrix correspond the sorted order of the\n"
      "segments.\n"
      "Usage: ivector-diarization-scoring [options] <reco2utt>"
      " <ivectors-rspecifier> <scores-wspecifier>\n"
      "e.g.: \n"
      "  ivector-diarization-scoring reco2utt scp:ivectors.scp"
      " ark:scores.ark ark,t:ivectors.1.ark\n";

    ParseOptions po(usage);
    BaseFloat target_energy = 0.5;
    bool convert_to_probs = false, use_cosing_scoring = true;
    std::string utt2num_frames_rspecifier;
    
    po.Register("target-energy", &target_energy,
      "Reduce dimensionality of i-vectors using PCA such that this fraction"
      " of the total energy remains.");
    po.Register("convert-to-probs", &convert_to_probs,
                "If specified, the cosine distances are transformed to "
                "probabilities.");
    po.Register("utt2num-frames-rspecifier", &utt2num_frames_rspecifier,
                "Weight i-vector by number of frames.");
    po.Register("use-cosine-scoring", &use_cosing_scoring,
                "Use cosine score metric instead of Euclidean norm.");
    KALDI_ASSERT(target_energy <= 1.0);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string reco2utt_rspecifier = po.GetArg(1),
      ivector_rspecifier = po.GetArg(2),
      scores_wspecifier = po.GetArg(3),
      out_reco2utt_wspecifier = po.GetArg(4);

    SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
    RandomAccessBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    BaseFloatMatrixWriter scores_writer(scores_wspecifier);
    TokenVectorWriter out_reco2utt_writer(out_reco2utt_wspecifier);
    RandomAccessInt32Reader utt2num_frames_reader(utt2num_frames_rspecifier);

    int32 num_reco_err = 0, num_reco_done = 0, num_utt_err = 0;
    for (; !reco2utt_reader.Done(); reco2utt_reader.Next()) {
      const std::string &reco = reco2utt_reader.Key();
      const std::vector<std::string> &uttlist = reco2utt_reader.Value();
      std::vector<std::string> out_uttlist;

      Matrix<BaseFloat> ivector_mat;
      Vector<BaseFloat> row_weights(uttlist.size());
      row_weights.Set(1.0);

      int32 id = 0;
      for (size_t i = 0; i < uttlist.size(); i++) {
        const std::string &utt = uttlist[i];

        if (!ivector_reader.HasKey(utt)) {
          KALDI_WARN << "No iVector present in input for utterance " << utt;
          num_utt_err++;
          continue;
        }

        const Vector<BaseFloat> &ivector = ivector_reader.Value(utt);

        if (ivector_mat.NumRows() == 0) {
          ivector_mat.Resize(uttlist.size(), ivector.Dim());
        }
      
        if (!utt2num_frames_rspecifier.empty()) {
          if (!utt2num_frames_reader.HasKey(utt)) {
            KALDI_WARN << "No weights present for utterance " << utt
                       << " in rspecifier " << utt2num_frames_rspecifier;
            num_utt_err++;
            continue;
          }

          row_weights(id) = utt2num_frames_reader.Value(utt);
        } else {
          row_weights(id) = 1.0;
        }

        ivector_mat.CopyRowFromVec(ivector, id);

        out_uttlist.push_back(utt);
        id++;
      }

      if (out_uttlist.size() == 0) {
        KALDI_WARN << "Not producing output for recording " << reco
                   << " since no segments had iVectors";
        num_reco_err++;
        continue;
      }
    
      ivector_mat.Resize(out_uttlist.size(), ivector_mat.NumCols(), kCopyData);
      row_weights.Resize(out_uttlist.size(), kCopyData);

      Matrix<BaseFloat> ivector_mat_pca, pca_transform,
        scores(out_uttlist.size(), out_uttlist.size());

      if (EstPca(ivector_mat, row_weights, target_energy, &pca_transform)) {
        // Apply PCA transform to the raw i-vectors.
        ApplyPca(ivector_mat, pca_transform, &ivector_mat_pca);
        KALDI_LOG << "For recording, " << reco << " retained " 
                  << ivector_mat_pca.NumCols() << " dimensions.";
      } else {
        KALDI_WARN << "Unable to compute conversation dependent PCA for"
                   << " recording " << reco << ".";
        ivector_mat_pca.Resize(ivector_mat.NumRows(), ivector_mat.NumCols());
        ivector_mat_pca.CopyFromMat(ivector_mat);
      }
   
      if (use_cosing_scoring) {
        // Compute dot-product between i-vectors
        scores.AddMatMat(1.0, ivector_mat_pca, kNoTrans, 
                         ivector_mat_pca, kTrans, 0.0);
      
        // Compute norms
        Vector<BaseFloat> norms(ivector_mat_pca.NumRows());
        for (int32 i = 0; i < ivector_mat_pca.NumRows(); i++) {
          norms(i) = ivector_mat_pca.Row(i).Norm(2);
        }

        // Compute cosine-score. Higher is better.
        // cosine score is between -1 and +1.
        // +1 is the best score.
        for (int32 i = 0; i < ivector_mat_pca.NumRows(); i++) {
          for (int32 j = 0; j < ivector_mat_pca.NumRows(); j++) {
            scores(i, j) /= (norms(i) * norms(j));
          }
        }
        
        // Convert to probabilities between 0 and 1. 
        // 1 is good. i.e. probability that two ivectors are same is 0.
        if (convert_to_probs) {
          scores.Scale(0.5);
          scores.Add(0.5);
        }
      } else {
        for (int32 i = 0; i < ivector_mat_pca.NumRows(); i++) {
          for (int32 j = i; j < ivector_mat_pca.NumRows(); j++) {
            Vector<BaseFloat> vec(ivector_mat_pca.Row(i));
            vec.AddVec(-1.0, ivector_mat_pca.Row(j));
            scores(i, j) = scores(j, i) = -VecVec(vec, vec);
          }
        }

        if (convert_to_probs)
          scores.ApplyExp();
      }

      scores_writer.Write(reco, scores);
      out_reco2utt_writer.Write(reco, out_uttlist);
      num_reco_done++;
    }
    KALDI_LOG << "Processed " << num_reco_done << " recordings, "
              << num_reco_err << " had errors; "
              << "failed with " << num_utt_err << " utterances.";
    return (num_reco_done != 0 ? 0 : 1 );
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
