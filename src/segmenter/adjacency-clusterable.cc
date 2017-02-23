// segmenter/adjacency-clusterable.cc

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

#include "segmenter/adjacency-clusterable.h"

namespace kaldi {

// Returns 1 if the segments (s1, e1) and (s2, e2) are adjacent.
BaseFloat AdjacencyScore(BaseFloat s1, BaseFloat e1, 
                         BaseFloat s2, BaseFloat e2) {
  if (s1 > s2) { 
    std::swap(s1, s2);
    std::swap(e1, e2);
  }

  // Segment 2 starts after end of segment 1 i.e. not adjacent.
  if (s2 > e1) return 0;
  return 1;
}
  
AdjacencyClusterable::AdjacencyClusterable(
    const std::set<int32> &points,
    const Vector<BaseFloat> *start_times,
    const Vector<BaseFloat> *end_times):
      points_(points), 
      start_times_(start_times), end_times_(end_times),
      objf_(0) {
  for (std::set<int32>::iterator itr_i = points_.begin();
       itr_i != points_.end(); ++itr_i) {
    for (std::set<int32>::iterator itr_j = itr_i;
         itr_j != points_.end(); ++itr_j) {
      if (itr_j == itr_i) {
        objf_ -= 1;
      } else {
        objf_ += AdjacencyScore(
            (*start_times)(*itr_i), (*end_times)(*itr_i),
            (*start_times)(*itr_j), (*end_times)(*itr_j));
      }
    }
  }
}

BaseFloat AdjacencyClusterable::Objf() const {
  return objf_;
}

void AdjacencyClusterable::SetZero() {
  points_.clear();
  objf_ = 0;
}

void AdjacencyClusterable::Add(const Clusterable &other_in) {
  const AdjacencyClusterable *other =
      static_cast<const AdjacencyClusterable*>(&other_in);

  objf_ += other->objf_;
  for (std::set<int32>::const_iterator itr_i = points_.begin();
       itr_i != points_.end(); ++itr_i) {
    for (std::set<int32>::const_iterator itr_j = other->points_.begin();
         itr_j != other->points_.end(); ++itr_j) {
      objf_ += AdjacencyScore(
          (*start_times_)(*itr_i), (*end_times_)(*itr_i),
          (*start_times_)(*itr_j), (*end_times_)(*itr_j));
    }
  }
  points_.insert(other->points_.begin(), other->points_.end());
}

void AdjacencyClusterable::Sub(const Clusterable &other_in) {
  const AdjacencyClusterable *other =
      static_cast<const AdjacencyClusterable*>(&other_in);
  for (std::set<int32>::const_iterator itr_i = points_.begin();
       itr_i != points_.end(); ++itr_i) {
    for (std::set<int32>::const_iterator itr_j = other->points_.begin();
         itr_j != other->points_.end(); ++itr_j) {
      objf_ -= AdjacencyScore(
          (*start_times_)(*itr_i), (*end_times_)(*itr_i),
          (*start_times_)(*itr_j), (*end_times_)(*itr_j));
    }
  }
  objf_ -= other->objf_;
  for (std::set<int32>::const_iterator itr_i = other->points_.begin();
       itr_i != other->points_.end(); ++itr_i) {
    points_.erase(*itr_i);
  }
}

BaseFloat AdjacencyClusterable::Normalizer() const {
  return points_.size();
}

Clusterable *AdjacencyClusterable::Copy() const {
  AdjacencyClusterable *ans = new AdjacencyClusterable(
      points_, start_times_, end_times_);
  return ans;
}

void AdjacencyClusterable::Scale(BaseFloat f) {
  // TODO
  return;
}

void AdjacencyClusterable::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "AdjCL");
  WriteToken(os, binary, "<Points>");
  std::vector<int32> vec(points_.begin(), points_.end());
  WriteIntegerVector(os, binary, vec);
  WriteToken(os, binary, "<Objf>");
  WriteBasicType(os, binary, objf_);
}

Clusterable *AdjacencyClusterable::ReadNew(std::istream &is, bool binary) const {
  // TODO
  return NULL;
}

BaseFloat AdjacencyClusterable::Distance(const Clusterable &other) const {
  Clusterable *copy = this->Copy();
  copy->Add(other);
  BaseFloat ans = (this->Objf() + other.Objf() - copy->Objf());
  delete copy;
  return ans;
}

}
