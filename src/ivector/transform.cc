// ivector/transform.cc

// Copyright 2013  Daniel Povey
//           2016  David Snyder
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

#include "ivector/transform.h"
#include "ivector/plda.h"

namespace kaldi {

bool EstPca(const Matrix<BaseFloat> &ivector_mat, 
            const Vector<BaseFloat> &row_weights,
            BaseFloat target_energy,
            Matrix<BaseFloat> *mat) {
  int32 num_rows = ivector_mat.NumRows(), num_cols = ivector_mat.NumCols();
  Vector<BaseFloat> sum(num_cols);
  SpMatrix<BaseFloat> sumsq(num_cols);
  sum.AddMatVec(1.0, ivector_mat, kTrans, row_weights, 0.0);
  sumsq.AddMat2Vec(1.0, ivector_mat, kTrans, row_weights, 0.0);
  sum.Scale(1.0 / row_weights.Sum());
  sumsq.Scale(1.0 / row_weights.Sum());
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
  KALDI_VLOG(2) << "Retained " << target_energy << " of total energy using "
                << dim << " dimensions.";
  return true;
}

void TransformIvectors(const Matrix<BaseFloat> &ivectors_in,
                       const PldaConfig &plda_config, const Plda &plda,
                       Matrix<BaseFloat> *ivectors_out) {
  int32 dim = plda.Dim();
  ivectors_out->Resize(ivectors_in.NumRows(), dim);
  for (int32 i = 0; i < ivectors_in.NumRows(); i++) {
    Vector<BaseFloat> transformed_ivector(dim);
    plda.TransformIvector(plda_config, ivectors_in.Row(i), 1.0,
                          &transformed_ivector);
    ivectors_out->Row(i).CopyFromVec(transformed_ivector);
  }
}

void TransformIvector(const Vector<BaseFloat> &ivector,
                      const PldaConfig &plda_config, const Plda &plda,
                      Vector<BaseFloat> *ivector_out) {
  int32 dim = plda.Dim();
  ivector_out->Resize(dim);
  plda.TransformIvector(plda_config, ivector, 1.0,
                        ivector_out);
}

void ApplyPca(const Matrix<BaseFloat> &ivector_mat,
              const Matrix<BaseFloat> &pca_mat, 
              Matrix<BaseFloat> *ivector_mat_out) {
  int32 transform_cols = pca_mat.NumCols(),
        transform_rows = pca_mat.NumRows(),
        feat_dim = ivector_mat.NumCols();
  ivector_mat_out->Resize(ivector_mat.NumRows(), transform_rows);
  KALDI_ASSERT(transform_cols == feat_dim);
  ivector_mat_out->AddMatMat(1.0, ivector_mat, kNoTrans,
                             pca_mat, kTrans, 0.0);
}

void ApplyPca(const Vector<BaseFloat> &ivector,
              const Matrix<BaseFloat> &pca_mat, 
              Vector<BaseFloat> *ivector_out) {
  int32 transform_cols = pca_mat.NumCols(),
        transform_rows = pca_mat.NumRows(),
        feat_dim = ivector.Dim();
  ivector_out->Resize(transform_rows);
  KALDI_ASSERT(transform_cols == feat_dim);
  ivector_out->AddMatVec(1.0, pca_mat, kNoTrans, ivector, 0.0);
}

}  // end namespace kaldi
