// segmenter/plda-clusterable.h

// Copyright 2017  Vimal Manohar

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

#ifndef KALDI_SEGMENTER_PLDA_CLUSTERABLE_H_
#define KALDI_SEGMENTER_PLDA_CLUSTERABLE_H_

#include <vector>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "itf/clusterable-itf.h"
#include "itf/options-itf.h"
#include "ivector/plda.h"

namespace kaldi {

struct PldaClusterableOptions {
  bool use_buggy_eqn;
  bool use_nfactor;
  bool normalize_distance;
  bool normalize_loglike;
  bool apply_logistic;
  bool use_avg_ivector;
  BaseFloat bic_penalty;

  PldaClusterableOptions() :
    use_buggy_eqn(false), use_nfactor(false), normalize_distance(false),
    normalize_loglike(false), apply_logistic(false), use_avg_ivector(false),
    bic_penalty(0.0) { }

  void Register(OptionsItf *opts);
};

class PldaClusterable: public PointsClusterable {
 public:
  PldaClusterable(): plda_(NULL), weight_(0.0), sumsq_(0.0) {}

  PldaClusterable(const PldaClusterableOptions &opts,
                  const Plda *plda):
    opts_(opts), plda_(plda), weight_(0.0), 
    stats_(plda->Dim()), sumsq_(0.0) { }

  PldaClusterable(const PldaClusterableOptions &opts,
                  const Plda *plda,
                  const std::set<int32> &points,
                  const Vector<BaseFloat> &vector,
                  BaseFloat weight);
  
  PldaClusterable(const Plda *plda,
                  const std::set<int32> &points,
                  const Vector<BaseFloat> &vector,
                  BaseFloat weight);

  virtual std::string Type() const {  return "plda"; }
  // Objf is the negated contribution to the PLDA log-likelihood ratio.
  virtual BaseFloat Objf() const;
  virtual void SetZero() { points_.clear(); weight_ = 0.0; 
                           sumsq_ = 0.0; stats_.Set(0.0); }
  virtual void AddStats(const VectorBase<BaseFloat> &vec,  
                        BaseFloat weight);
  virtual void Add(const Clusterable &other_in);
  virtual void Sub(const Clusterable &other_in);
  virtual BaseFloat Normalizer() const { return weight_; }
  virtual Clusterable *Copy() const;
  virtual void Scale(BaseFloat f);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Clusterable *ReadNew(std::istream &is, bool binary) const;
  virtual ~PldaClusterable() {}
  virtual BaseFloat Distance(const Clusterable &other) const;
  
  const std::set<int32> &points() const { return points_; }
  const Plda* plda() const { return plda_; }

  PldaClusterableOptions opts_;

 private:
  const Plda* plda_;  // Pointer to Plda object
  std::set<int32> points_;
  double weight_;  // sum of weights of the source vectors.  Never negative.
  Vector<double> stats_; // Equals the weighted sum of the source vectors.
                  
  double sumsq_;  // Equals the sum over all sources, of weight_ * vec.vec,
                  // where vec = stats_ / weight_.  Used in computing
                  // the objective function.
  
  void Read(std::istream &is, bool binary);
};

}  // end namespace kaldi

#endif  // KALDI_SEGMENTER_PLDA_CLUSTERABLE_H_
