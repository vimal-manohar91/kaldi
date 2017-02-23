// segmenter/adjacency-clusterable.h

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

#ifndef KALDI_SEGMENTER_ADJACENCY_CLUSTERABLE_H_
#define KALDI_SEGMENTER_ADJACENCY_CLUSTERABLE_H_

#include <vector>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "itf/clusterable-itf.h"

namespace kaldi {

class AdjacencyClusterable: public Clusterable {
 public:
  AdjacencyClusterable(const std::set<int32> &points,
                       const Vector<BaseFloat> *start_times,
                       const Vector<BaseFloat> *end_times);
  virtual std::string Type() const { return "adj"; }
  virtual BaseFloat Objf() const;
  virtual void SetZero();
  virtual void Add(const Clusterable &other_in);
  virtual void Sub(const Clusterable &other_in);
  virtual BaseFloat Normalizer() const;
  virtual Clusterable *Copy() const;
  virtual void Scale(BaseFloat f);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Clusterable *ReadNew(std::istream &is, bool binary) const;
  virtual ~AdjacencyClusterable() {}

  virtual BaseFloat Distance(const Clusterable &other) const;

  virtual std::ostream& operator<< (std::ostream &os) const {
    Write(os, false);
    return os;
  }

  virtual const std::set<int32> &points() const { return points_; }
  virtual const Vector<BaseFloat>* start_times() const { return start_times_; }
  virtual const Vector<BaseFloat>* end_times() const { return end_times_; }

 private:
  std::set<int32> points_;
  const Vector<BaseFloat> *start_times_;
  const Vector<BaseFloat> *end_times_; 
  BaseFloat objf_;
};

}

#endif  // KALDI_SEGMENTER_ADJACENCY_CLUSTERABLE_H_

