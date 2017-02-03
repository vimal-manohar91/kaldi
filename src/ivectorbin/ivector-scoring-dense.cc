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
#include "ivector/ivector-clusterable.h"

namespace kaldi {

void ApplySigmoid(Matrix<BaseFloat> *mat) {
  mat->Scale(-1.0);
  mat->ApplyExp();
  for (int32 i = 0; i < mat->NumRows(); i++) {
    for (int32 j = 0; j < mat->NumCols(); j++) {
      (*mat)(i,j) = 1.0 / (*mat)(i, j);
    }
  }
}

bool EstPca(const Matrix<BaseFloat> &ivector_mat, BaseFloat target_energy,
  Matrix<BaseFloat> *mat) {
  int32 num_rows = ivector_mat.NumRows(),
    num_cols = ivector_mat.NumCols();
  Vector<BaseFloat> sum;
  SpMatrix<BaseFloat> sumsq;
  sum.Resize(num_cols);
  sumsq.Resize(num_cols);
  sum.AddRowSumMat(1.0, ivector_mat);
  sumsq.AddMat2(1.0, ivector_mat, kTrans, 1.0);
  sum.Scale(1.0 / num_rows);
  sumsq.Scale(1.0 / num_rows);
  sumsq.AddVec2(-1.0, sum); // now sumsq is centered covariance.
  int32 full_dim = sum.Dim();

  Matrix<BaseFloat> P(full_dim, full_dim);
  Vector<BaseFloat> s(full_dim);

  try {
    if (num_rows > num_cols)
      sumsq.Eig(&s, &P);
    else
      Matrix<BaseFloat>(sumsq).Svd(&s, &P, NULL);
  } catch (...) {
    return false;
  }

  SortSvd(&s, &P);

  Matrix<BaseFloat> transform(P, kTrans); // Transpose of P.  This is what
                                       // appears in the transform.
  Vector<BaseFloat> offset(full_dim);

  // We want the PCA transform to retain target_energy amount of the total
  // energy.
  BaseFloat total_energy = s.Sum();
  BaseFloat energy = 0.0;
  int32 dim = 1;
  while (energy / total_energy <= target_energy) {
    energy += s(dim-1);
    dim++;
  }
  Matrix<BaseFloat> transform_float(transform);
  mat->Resize(transform.NumCols(), transform.NumRows());
  mat->CopyFromMat(transform);
  mat->Resize(dim, transform_float.NumCols(), kCopyData);
  return true;
}

void ApplyPca(const Matrix<BaseFloat> &ivector_mat,
  const Matrix<BaseFloat> &pca_mat, Matrix<BaseFloat> *ivector_mat_out) {

  int32 transform_cols = pca_mat.NumCols(),
        transform_rows = pca_mat.NumRows(),
        feat_dim = ivector_mat.NumCols();
  ivector_mat_out->Resize(ivector_mat.NumRows(), transform_rows);
  KALDI_ASSERT(transform_cols == feat_dim);
  ivector_mat_out->AddMatMat(1.0, ivector_mat, kNoTrans,
    pca_mat, kTrans, 0.0);
}

}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Perform cosine scoring for speaker diarization.  The input spk2utt\n"
      "should be of the form <recording-id> <seg1> <seg2> ... <segN> and\n"
      "there should be one iVector for each segment.  Cosine scoring is\n"
      "performed between all pairs of iVectors in a recording and outputs\n"
      "an archive of score matrices, one for each recording-id.  The rows\n"
      "and columns of the the matrix correspond the sorted order of the\n"
      "segments.\n"
      "Usage: ivector-diarization-scoring [options] <spk2utt>"
      " <ivectors-rspecifier> <scores-wspecifier>\n"
      "e.g.: \n"
      "  ivector-diarization-scoring spk2utt scp:ivectors.scp"
      " ark:scores.ark ark,t:ivectors.1.ark\n";

    ParseOptions po(usage);
    BaseFloat target_energy = 0.5;

    po.Register("target-energy", &target_energy,
      "Reduce dimensionality of i-vectors using PCA such that this fraction"
      " of the total energy remains.");
    KALDI_ASSERT(target_energy <= 1.0);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string spk2utt_rspecifier = po.GetArg(1),
      ivector_rspecifier = po.GetArg(2),
      scores_wspecifier = po.GetArg(3);

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    BaseFloatMatrixWriter scores_writer(scores_wspecifier);
    int32 num_spk_err = 0,
          num_spk_done = 0;
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();

      // The uttlist is sorted here and in binaries that use the scores
      // this outputs.  This is to ensure that the segment corresponding
      // to the same rows and columns (of the score matrix) across binaries.
      std::vector<std::string> uttlist = spk2utt_reader.Value();
      std::sort(uttlist.begin(), uttlist.end());
      std::vector<Vector<BaseFloat> > ivectors;

      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];

        if (!ivector_reader.HasKey(utt)) {
          KALDI_ERR << "No iVector present in input for utterance " << utt;
        }

        Vector<BaseFloat> ivector = ivector_reader.Value(utt);
        ivectors.push_back(ivector);
      }
      if (ivectors.size() == 0) {
        KALDI_WARN << "Not producing output for recording " << spk
                   << " since no segments had iVectors";
        num_spk_err++;
      } else {
        Matrix<BaseFloat> ivector_mat(ivectors.size(), ivectors[0].Dim()),
                          ivector_mat_pca,
                          pca_transform,
                          scores(ivectors.size(), ivectors.size());

        for (size_t i = 0; i < ivectors.size(); i++) {
          ivector_mat.Row(i).CopyFromVec(ivectors[i]);
        }
        if (EstPca(ivector_mat, target_energy, &pca_transform)) {
          // Apply PCA transform to the raw i-vectors.
          ApplyPca(ivector_mat, pca_transform, &ivector_mat_pca);
        } else {
          KALDI_WARN << "Unable to compute conversation dependent PCA for"
            << " recording " << spk << ".";
          ivector_mat_pca.Resize(ivector_mat.NumRows(), ivector_mat.NumCols());
          ivector_mat_pca.CopyFromMat(ivector_mat);
        }
        
        scores.AddMatMat(1.0, ivector_mat_pca, kNoTrans, 
                         ivector_mat_pca, kTrans, 0.0);
        
        Vector<BaseFloat> norms(ivector_mat_pca.NumRows());
        for (int32 i = 0; i < ivector_mat_pca.NumRows(); i++) {
          norms(i) = ivector_mat_pca.Row(i).Norm(2);
        }
        
        for (int32 i = 0; i < ivector_mat_pca.NumRows(); i++) {
          for (int32 j = 0; j < ivector_mat_pca.NumRows(); j++) {
            scores(i, j) /= -(norms(i) * norms(j));
          }
        }

        scores.Scale(0.5);
        scores.Add(0.5);
        
        //Sigmoid(scores);

        //    scores(i, j) /= 
        //    IvectorClusterable c1(Vector<BaseFloat>(ivector_mat_pca.Row(i)), 1.0);
        //    IvectorClusterable c2(Vector<BaseFloat>(ivector_mat_pca.Row(j)), 1.0);
        //    scores(i, j) = c1.Distance(c2);
        //    ///scores(i,j) = -VecVec(Vector<double>(ivector_mat_pca.Row(i)),
        //    ///                      Vector<double>(ivector_mat_pca.Row(j)))
        //    ///            / ivector_mat_pca.Row(i).Norm(2) 
        //    ///            / ivector_mat_pca.Row(j).Norm(2);
        //  }
        //}
        scores_writer.Write(spk, scores);
        num_spk_done++;
      }
    }
    KALDI_LOG << "Processed " << num_spk_done << " recordings, "
              << num_spk_err << " had errors.";
    return (num_spk_done != 0 ? 0 : 1 );
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

