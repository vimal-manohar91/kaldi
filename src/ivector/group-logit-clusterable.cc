// ivector/group-logit-clusterable.cc

// Copyright 2016  David Snyder

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


#include "ivector/group-logit-clusterable.h"

namespace kaldi {

double Sigmoid(double x) {
  if (x > 0.0) {
    x = 1.0 / (1.0 + Exp(-x));
  } else {
    double ex = Exp(x);
    x = ex / (ex + 1.0);
  }
  return x;
}

BaseFloat GroupLogitClusterable::Objf() const {
  // TODO (current only using Distance())
  return total_distance_;
}

void GroupLogitClusterable::SetZero() {
  points_.clear();
  total_distance_ = 0;
}

void GroupLogitClusterable::Add(const Clusterable &other_in) {
  const GroupLogitClusterable *other =
      static_cast<const GroupLogitClusterable*>(&other_in);
  double sum = (total_distance_);
  for (std::set<int32>::const_iterator itr_i = points_.begin();
       itr_i != points_.end(); ++itr_i) {
    for (std::set<int32>::const_iterator itr_j = other->points_.begin();
         itr_j != other->points_.end(); ++itr_j) {
      sum += ((*scores_)(*itr_i, *itr_j));
    }
  }
  sum += (other->total_distance_);
  total_distance_ = sum;
  points_.insert(other->points_.begin(), other->points_.end());
}

void GroupLogitClusterable::Sub(const Clusterable &other_in) {
  const GroupLogitClusterable *other =
      static_cast<const GroupLogitClusterable*>(&other_in);
  double sum = (total_distance_);
  for (std::set<int32>::const_iterator itr_i = points_.begin();
       itr_i != points_.end(); ++itr_i) {
    for (std::set<int32>::const_iterator itr_j = other->points_.begin();
         itr_j != other->points_.end(); ++itr_j) {
      sum -= ((*scores_)(*itr_i, *itr_j));
    }
  }
  sum -= (other->total_distance_);
  total_distance_ = (sum);
  for (std::set<int32>::const_iterator itr_i = other->points_.begin();
       itr_i != other->points_.end(); ++itr_i) {
    points_.erase(*itr_i);
  }
}

BaseFloat GroupLogitClusterable::Normalizer() const {
  return points_.size();
}

Clusterable *GroupLogitClusterable::Copy() const {
  GroupLogitClusterable *ans = new GroupLogitClusterable(points_, scores_);
  return ans;
}

void GroupLogitClusterable::Scale(BaseFloat f) {
  // TODO
  return;
}

void GroupLogitClusterable::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "GrCL");
  std::vector<int32> vec(points_.begin(), points_.end());
  WriteIntegerVector(os, binary, vec);
  WriteBasicType(os, binary, total_distance_);
}

Clusterable *GroupLogitClusterable::ReadNew(std::istream &is, bool binary) const {
  // TODO
  return NULL;
}

BaseFloat GroupLogitClusterable::Distance(const Clusterable &other_in) const {
  const GroupLogitClusterable *other =
      static_cast<const GroupLogitClusterable*>(&other_in);
  GroupLogitClusterable *copy = static_cast<GroupLogitClusterable*>(this->Copy());
  copy->Add(*other);
  BaseFloat ans = (copy->Objf() - other->Objf() - this->Objf())
                   / (other->Normalizer() * this->Normalizer());
  return Sigmoid(ans);
}
}
