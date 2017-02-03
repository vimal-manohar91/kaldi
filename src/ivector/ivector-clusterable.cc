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
#include "ivector/ivector-clusterable.h"

namespace kaldi {
  
void IvectorClusterable::Add(const Clusterable &other_in) {
  KALDI_ASSERT(other_in.Type() == "ivector");
  const IvectorClusterable *other =
      static_cast<const IvectorClusterable*>(&other_in);
  weight_ += other->weight_;
  stats_.AddVec(1.0, other->stats_);
}

void IvectorClusterable::Sub(const Clusterable &other_in) {
  KALDI_ASSERT(other_in.Type() == "ivector");
  const IvectorClusterable *other =
      static_cast<const IvectorClusterable*>(&other_in);
  weight_ -= other->weight_;
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
    stats_.Set(0.0);
  }
}

Clusterable* IvectorClusterable::Copy() const {
  IvectorClusterable *ans = new IvectorClusterable();
  ans->weight_ = weight_;
  ans->stats_ = stats_;
  return ans;
}

void IvectorClusterable::Scale(BaseFloat f) {
  KALDI_ASSERT(f >= 0.0);
  weight_ *= f;
  stats_.Scale(f);
}

void IvectorClusterable::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "iVCL");  // magic string.
  WriteToken(os, binary, "<Weight>");
  WriteBasicType(os, binary, weight_);
  WriteToken(os, binary, "<Stats>");    
  stats_.Write(os, binary);
}

Clusterable* IvectorClusterable::ReadNew(std::istream &is, bool binary) const {
  IvectorClusterable *vc = new IvectorClusterable();
  vc->Read(is, binary);
  return vc;
}

void IvectorClusterable::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "iVCL");  // magic string.
  ExpectToken(is, binary, "<Weight>");
  ReadBasicType(is, binary, &weight_);
  ExpectToken(is, binary, "<Stats>");    
  stats_.Read(is, binary);
}

IvectorClusterable::IvectorClusterable(const Vector<BaseFloat> &vector,
                                     BaseFloat weight):
    weight_(weight), stats_(vector) {
  stats_.Scale(weight);
  KALDI_ASSERT(weight >= 0.0);
}    

BaseFloat IvectorClusterable::Objf() const {
  double direct_sumsq;
  if (weight_ > std::numeric_limits<BaseFloat>::min()) {
    direct_sumsq = VecVec(stats_, stats_) / weight_;
  } else {
    direct_sumsq = 0.0;
  }
  return direct_sumsq;
}

BaseFloat IvectorClusterable::Distance(const Clusterable &other_in) const {
  KALDI_ASSERT(other_in.Type() == "ivector");
  const IvectorClusterable *other = 
    static_cast<const IvectorClusterable*>(&other_in);
    
  Vector<double> vec1(stats_);
  vec1.Scale(1.0 / weight_);

  Vector<double> vec2(other->stats_);
  vec2.Scale(1.0 / other->weight_);

  return -0.5 * VecVec(vec1, vec2) / vec1.Norm(2) / vec2.Norm(2) + 0.5;
  //return 1.0 / (1.0 + Exp(VecVec(vec1, vec2) / vec1.Norm(2) / vec2.Norm(2)));
}

}  // end namespace kaldi
