// ivector/ivector-clusterable.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University
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

#ifndef KALDI_SEGMENTER_IVECTOR_CLUSTERABLE_H_
#define KALDI_SEGMENTER_IVECTOR_CLUSTERABLE_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "itf/options-itf.h"
#include "itf/clusterable-itf.h"

namespace kaldi {

struct IvectorClusterableOptions {
  bool use_cosine_distance;

  IvectorClusterableOptions() :
    use_cosine_distance(false) { }

  void Register(OptionsItf *opts);
};

class IvectorClusterable: public PointsClusterable {
 public:
  IvectorClusterable(): weight_(0.0), sumsq_(0.0) {}

  IvectorClusterable(const IvectorClusterableOptions &opts):
    opts_(opts), weight_(0.0), sumsq_(0.0) { }
                     
  IvectorClusterable(const IvectorClusterableOptions &opts,
                     const std::set<int32> &points,
                     const Vector<BaseFloat> &vector,
                     BaseFloat weight);
  
  IvectorClusterable(const std::set<int32> &points,
                     const Vector<BaseFloat> &vector,
                     BaseFloat weight);

  virtual std::string Type() const {  return "ivector"; }
  // Objf is negated weighted sum of squared distances.
  virtual BaseFloat Objf() const;
  virtual void SetZero() { points_.clear(); weight_ = 0.0; 
                           sumsq_ = 0.0; stats_.Set(0.0); }
  virtual void Add(const Clusterable &other_in);
  virtual void Sub(const Clusterable &other_in);
  virtual BaseFloat Normalizer() const { return weight_; }
  virtual Clusterable *Copy() const;
  virtual void Scale(BaseFloat f);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Clusterable *ReadNew(std::istream &is, bool binary) const;
  virtual ~IvectorClusterable() {}
  virtual BaseFloat Distance(const Clusterable &other) const;

  const std::set<int32>& points() const { return points_; }

  IvectorClusterableOptions opts_;

 private:
  std::set<int32> points_;
  double weight_;  // sum of weights of the source vectors.  Never negative.
  Vector<double> stats_; // Equals the weighted sum of the source vectors.
                  
  double sumsq_;  // Equals the sum over all sources, of weight_ * vec.vec,
                  // where vec = stats_ / weight_.  Used in computing
                  // the objective function.
  
  void Read(std::istream &is, bool binary);
};

}  // end namespace kaldi

#endif  // KALDI_SEGMENTER_IVECTOR_CLUSTERABLE_H_
