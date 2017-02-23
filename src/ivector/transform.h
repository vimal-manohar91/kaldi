// ivector/transform.h

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


#ifndef KALDI_IVECTOR_TRANSFORM_H_
#define KALDI_IVECTOR_TRANSFORM_H_

#include "ivector/plda.h"

namespace kaldi {

bool EstPca(const Matrix<BaseFloat> &ivector_mat, 
            const Vector<BaseFloat> &row_weights,
            BaseFloat target_energy,
            Matrix<BaseFloat> *mat);

void TransformIvectors(const Matrix<BaseFloat> &ivectors_in,
                       const PldaConfig &plda_config, const Plda &plda,
                       Matrix<BaseFloat> *ivectors_out);

void TransformIvector(const Vector<BaseFloat> &ivector,
                      const PldaConfig &plda_config, const Plda &plda,
                      Vector<BaseFloat> *ivector_out);

void ApplyPca(const Matrix<BaseFloat> &ivector_mat,
              const Matrix<BaseFloat> &pca_mat, 
              Matrix<BaseFloat> *ivector_mat_out);

void ApplyPca(const Vector<BaseFloat> &ivector,
              const Matrix<BaseFloat> &pca_mat, 
              Vector<BaseFloat> *ivector_out);

}

#endif  // KALDI_IVECTOR_TRANSFORM_H_
