// ivector/ivector-clusterable.h

// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//           2016       Vimal Manohar

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
#include "itf/clusterable-itf.h"
#include "segmenter/ivector-clusterable.h"

namespace kaldi {

void IvectorClusterableOptions::Register(OptionsItf *opts) {
  opts->Register("use-cosine-distance", &use_cosine_distance,
                 "Use cosine distance");
}

void IvectorClusterable::Add(const Clusterable &other_in) {
  KALDI_ASSERT(other_in.Type() == "ivector");
  const IvectorClusterable *other =
      static_cast<const IvectorClusterable*>(&other_in);
  weight_ += other->weight_;
  stats_.AddVec(1.0, other->stats_);
  sumsq_ += other->sumsq_;
  
  points_.insert(other->points_.begin(), other->points_.end());
}

void IvectorClusterable::Sub(const Clusterable &other_in) {
  KALDI_ASSERT(other_in.Type() == "ivector");
  const IvectorClusterable *other =
      static_cast<const IvectorClusterable*>(&other_in);
  weight_ -= other->weight_;
  sumsq_ -= other->sumsq_;
  stats_.AddVec(-1.0, other->stats_);
  if (weight_ < 0.0) {
    if (weight_ < -0.1 && weight_ < -0.0001 * fabs(other->weight_)) {
      // a negative weight may indicate an algorithmic error if it is
      // encountered.
      KALDI_WARN << "Negative weight encountered " << weight_;
    }
    weight_ = 0.0;
  }
  if (weight_ == 0.0) {
    sumsq_ = 0.0;
    stats_.Set(0.0);
  }

  for (std::set<int32>::const_iterator itr_i = other->points_.begin();
       itr_i != other->points_.end(); ++itr_i) {
    points_.erase(*itr_i);
  }
}

Clusterable* IvectorClusterable::Copy() const {
  IvectorClusterable *ans = new IvectorClusterable(opts_);
  ans->weight_ = weight_;
  ans->sumsq_ = sumsq_;
  ans->stats_ = stats_;
  ans->points_ = points_;
  return ans;
}

void IvectorClusterable::Scale(BaseFloat f) {
  KALDI_ASSERT(f >= 0.0);
  weight_ *= f;
  stats_.Scale(f);
  sumsq_ *= f;
}

void IvectorClusterable::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "iVCL");  // magic string.
  std::vector<int32> vec(points_.begin(), points_.end());
  WriteIntegerVector(os, binary, vec);
  WriteToken(os, binary, "<Weight>");
  WriteBasicType(os, binary, weight_);
  WriteToken(os, binary, "<Sumsq>");  
  WriteBasicType(os, binary, sumsq_);
  WriteToken(os, binary, "<Stats>");    
  stats_.Write(os, binary);
}

Clusterable* IvectorClusterable::ReadNew(std::istream &is, bool binary) const {
  IvectorClusterable *vc = new IvectorClusterable(opts_);
  vc->Read(is, binary);
  return vc;
}

void IvectorClusterable::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "iVCL");  // magic string.
  std::vector<int32> vec;
  ReadIntegerVector(is, binary, &vec);
  points_.clear();
  std::copy(vec.begin(), vec.end(), std::inserter(points_, points_.end()));
  ExpectToken(is, binary, "<Weight>");
  ReadBasicType(is, binary, &weight_);
  ExpectToken(is, binary, "<Sumsq>");  
  ReadBasicType(is, binary, &sumsq_);
  ExpectToken(is, binary, "<Stats>");    
  stats_.Read(is, binary);
}

IvectorClusterable::IvectorClusterable(const IvectorClusterableOptions &opts,
                                       const std::set<int32> &points,
                                       const Vector<BaseFloat> &vector,
                                       BaseFloat weight):
    opts_(opts), points_(points), weight_(weight), stats_(vector), 
    sumsq_(0.0) {
  stats_.Scale(weight);
  KALDI_ASSERT(weight >= 0.0);
  sumsq_ = VecVec(vector, vector) * weight;
}

IvectorClusterable::IvectorClusterable(const std::set<int32> &points,
                                       const Vector<BaseFloat> &vector,
                                       BaseFloat weight):
    points_(points), weight_(weight), stats_(vector), sumsq_(0.0) {
  stats_.Scale(weight);
  KALDI_ASSERT(weight >= 0.0);
  sumsq_ = VecVec(vector, vector) * weight;
}

BaseFloat IvectorClusterable::Objf() const {
  double direct_sumsq;
  if (weight_ > std::numeric_limits<BaseFloat>::min()) {
    direct_sumsq = VecVec(stats_, stats_) / weight_;
  } else {
    direct_sumsq = 0.0;
  }
  // ans is a negated weighted sum of squared distances; it should not be
  // positive.
  double ans = -(sumsq_ - direct_sumsq); 
  if (ans > 0.0) {
    if (ans > 1.0) {
      KALDI_WARN << "Positive objective function encountered (treating as zero): "
                 << ans;
    }
    ans = 0.0;
  }
  
  return ans;
}

BaseFloat IvectorClusterable::Distance(const Clusterable &other_in) const {
  KALDI_ASSERT(other_in.Type() == "ivector");
  const IvectorClusterable *other = 
    static_cast<const IvectorClusterable*>(&other_in);
    
  if (opts_.use_cosine_distance) {
    Vector<double> vec1(stats_);
    vec1.Scale(1.0 / weight_);

    Vector<double> vec2(other->stats_);
    vec2.Scale(1.0 / other->weight_);

    return -0.5 * VecVec(vec1, vec2) / vec1.Norm(2) / vec2.Norm(2) + 0.5;
    //return 1.0 / (1.0 + Exp(VecVec(vec1, vec2) / vec1.Norm(2) / vec2.Norm(2)));
  }

  Clusterable *copy = static_cast<IvectorClusterable*>(this->Copy());
  copy->Add(*other);

  BaseFloat ans = this->Objf() + other->Objf() - copy->Objf();

  delete copy;
  return ans;
}

}  // end namespace kaldi
