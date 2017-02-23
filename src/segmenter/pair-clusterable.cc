// segmenter/pair-clusterable.cc

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

#include "segmenter/pair-clusterable.h"

namespace kaldi {

BaseFloat PairClusterable::Objf() const {
  return weight1_ * clusterable1_->Objf() 
    + weight2_ * clusterable2_->Objf();
}
  
void PairClusterable::SetZero() { 
  clusterable1_->SetZero();
  clusterable2_->SetZero();
}

void PairClusterable::Add(const Clusterable &other_in) {
  KALDI_ASSERT(other_in.Type() == "pair");
  const PairClusterable *other =
      static_cast<const PairClusterable*>(&other_in);
  clusterable1_->Add(*(other->clusterable1_));
  clusterable2_->Add(*(other->clusterable2_));
}

void PairClusterable::Sub(const Clusterable &other_in) {
  KALDI_ASSERT(other_in.Type() == "pair");
  const PairClusterable *other =
      static_cast<const PairClusterable*>(&other_in);
  clusterable1_->Sub(*(other->clusterable1_));
  clusterable2_->Sub(*(other->clusterable2_));
}

Clusterable* PairClusterable::Copy() const {
  PairClusterable *ans = new PairClusterable(
      clusterable1_->Copy(), clusterable2_->Copy(),
      weight1_, weight2_);
  return ans;
}

void PairClusterable::Scale(BaseFloat f) {
  // TODO
  return;
}

void PairClusterable::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "PairCL");  // magic string.
  clusterable1_->Write(os, binary);
  clusterable2_->Write(os, binary);
}

Clusterable* PairClusterable::ReadNew(std::istream &is, bool binary) const {
  KALDI_ERR << "Cannot be implemented!";
  return NULL;
}

BaseFloat PairClusterable::Distance(const Clusterable &other_in) const {
  KALDI_ASSERT(other_in.Type() == "pair");
  const PairClusterable *other =
      static_cast<const PairClusterable*>(&other_in);

  return weight1_ * clusterable1_->Distance(*(other->clusterable1_))
    + weight2_ * clusterable2_->Distance(*(other->clusterable2_));
}

BaseFloat PairClusterable::MergeThreshold(const Clusterable &other_in) const {
  KALDI_ASSERT(other_in.Type() == "pair");
  const PairClusterable *other =
      static_cast<const PairClusterable*>(&other_in);
  return clusterable1_->Distance(*(other->clusterable1_));
}

}  // end namespace kaldi
