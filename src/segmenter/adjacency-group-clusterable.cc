// segmenter/adjacency-group-clusterable.cc

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

#include "segmenter/adjacency-group-clusterable.h"

namespace kaldi {

BaseFloat AdjacencyScore(BaseFloat s1, BaseFloat e1, 
                         BaseFloat s2, BaseFloat e2) {
  if (s1 < s2) { 
    std::swap(s1, s2);
    std::swap(e1, e2);
  }

  if (s2 > e1) return 0;
  return 1;
}
  
AdjacencyGroupClusterable::AdjacencyGroupClusterable(
    const std::set<int32> &points,
    const Matrix<BaseFloat> *scores,
    const Vector<BaseFloat> *start_times,
    const Vector<BaseFloat> *end_times,
    BaseFloat adjacency_factor):
      points_(points), scores_(scores),
      start_times_(start_times), end_times_(end_times),
      total_distance_(0), adjacency_score_(0), 
      adjacency_factor_(adjacency_factor) {
  for (std::set<int32>::iterator itr_i = points_.begin();
    itr_i != points_.end(); ++itr_i) {
    for (std::set<int32>::iterator itr_j = itr_i;
      itr_j != points_.end(); ++itr_j) {
      total_distance_ += (*scores_)(*itr_i, *itr_j);
      adjacency_score_ += 1 - AdjacencyScore(
          (*start_times)(*itr_i), (*end_times)(*itr_i),
          (*start_times)(*itr_j), (*end_times)(*itr_j));
    }
  }
}

BaseFloat AdjacencyGroupClusterable::Objf() const {
  // TODO (current only using Distance())
  return total_distance_ + adjacency_factor_ * adjacency_score_;
}

void AdjacencyGroupClusterable::SetZero() {
  points_.clear();
  total_distance_ = 0;
  adjacency_score_ = 0;
}

void AdjacencyGroupClusterable::Add(const Clusterable &other_in) {
  const AdjacencyGroupClusterable *other =
      static_cast<const AdjacencyGroupClusterable*>(&other_in);

  adjacency_score_ += other->adjacency_score_;
  for (std::set<int32>::const_iterator itr_i = points_.begin();
       itr_i != points_.end(); ++itr_i) {
    for (std::set<int32>::const_iterator itr_j = other->points_.begin();
         itr_j != other->points_.end(); ++itr_j) {
      total_distance_ += (*scores_)(*itr_i, *itr_j);

      adjacency_score_ -= AdjacencyScore(
          (*start_times_)(*itr_i), (*end_times_)(*itr_i),
          (*start_times_)(*itr_j), (*end_times_)(*itr_j));
    }
  }
  total_distance_ += other->total_distance_;
  points_.insert(other->points_.begin(), other->points_.end());
}

void AdjacencyGroupClusterable::Sub(const Clusterable &other_in) {
  const AdjacencyGroupClusterable *other =
      static_cast<const AdjacencyGroupClusterable*>(&other_in);
  for (std::set<int32>::const_iterator itr_i = points_.begin();
       itr_i != points_.end(); ++itr_i) {
    for (std::set<int32>::const_iterator itr_j = other->points_.begin();
         itr_j != other->points_.end(); ++itr_j) {
      total_distance_ -= (*scores_)(*itr_i, *itr_j);
      adjacency_score_ += AdjacencyScore(
          (*start_times_)(*itr_i), (*end_times_)(*itr_i),
          (*start_times_)(*itr_j), (*end_times_)(*itr_j));
    }
  }
  adjacency_score_ -= other->adjacency_score_;
  total_distance_ -= other->total_distance_;
  for (std::set<int32>::const_iterator itr_i = other->points_.begin();
       itr_i != other->points_.end(); ++itr_i) {
    points_.erase(*itr_i);
  }
}

BaseFloat AdjacencyGroupClusterable::Normalizer() const {
  return points_.size();
}

Clusterable *AdjacencyGroupClusterable::Copy() const {
  AdjacencyGroupClusterable *ans = new AdjacencyGroupClusterable(
      points_, scores_, start_times_, end_times_, adjacency_factor_);
  return ans;
}

void AdjacencyGroupClusterable::Scale(BaseFloat f) {
  // TODO
  return;
}

void AdjacencyGroupClusterable::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "AdjGrCL");
  std::vector<int32> vec(points_.begin(), points_.end());
  WriteIntegerVector(os, binary, vec);
  WriteBasicType(os, binary, total_distance_);
  WriteBasicType(os, binary, adjacency_score_);
}

Clusterable *AdjacencyGroupClusterable::ReadNew(std::istream &is, bool binary) const {
  // TODO
  return NULL;
}

BaseFloat AdjacencyGroupClusterable::Distance(const Clusterable &other_in) const {
  const AdjacencyGroupClusterable *other =
      static_cast<const AdjacencyGroupClusterable*>(&other_in);
  AdjacencyGroupClusterable *copy = static_cast<AdjacencyGroupClusterable*>(
      this->Copy());
  copy->Add(*other);
  BaseFloat ans = (copy->total_distance_ - other->total_distance_ 
                   - this->total_distance_)
                   / (other->Normalizer() * this->Normalizer());

  // If the clusters are adjacent the adjacency_score of the copy is lower 
  // than the sum and hence the distance is reduced.
  ans += adjacency_factor_ * (copy->adjacency_score_ - other->adjacency_score_
                              - this->adjacency_score_);
  return std::max(ans, 0.0f);
}

}

