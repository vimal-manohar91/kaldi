// ivector/group-clusterable.h

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

#ifndef KALDI_IVECTOR_GROUP_CLUSTERABLE_H_
#define KALDI_IVECTOR_GROUP_CLUSTERABLE_H_

#include <vector>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "itf/clusterable-itf.h"
#include "itf/options-itf.h"

namespace kaldi {

struct GroupClusterableOptions {
  bool normalize_by_count;

  GroupClusterableOptions():
    normalize_by_count(true) { }

  void Register(OptionsItf *opts) {
    opts->Register("normalize-by-count", &normalize_by_count,
                   "Normalize the distance by the "
                   "total number of points in the two clusters");
  }
};

class GroupClusterable: public Clusterable {
 public:
  GroupClusterable(const std::set<int32> &points,
                   const Matrix<BaseFloat> *scores);
  
  GroupClusterable(const GroupClusterableOptions &opts,
                   const std::set<int32> &points,
                   const Matrix<BaseFloat> *scores);

  virtual std::string Type() const { return "group"; }
  virtual BaseFloat Objf() const;
  virtual void SetZero();
  virtual void Add(const Clusterable &other_in);
  virtual void Sub(const Clusterable &other_in);
  virtual BaseFloat Normalizer() const;
  virtual Clusterable *Copy() const;
  virtual void Scale(BaseFloat f);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Clusterable *ReadNew(std::istream &is, bool binary) const;
  virtual ~GroupClusterable() {}
  virtual BaseFloat Distance(const Clusterable &other_in) const;
  virtual std::ostream& operator<< (std::ostream &os) const {
    std::vector<int32> vec(points_.begin(), points_.end());
    WriteIntegerVector(os, false, vec);
    return os;
  }

  virtual const std::set<int32> &points() { return points_; }
  virtual const Matrix<BaseFloat> *scores() { return scores_; }

 private:
  GroupClusterableOptions opts_;

  std::set<int32> points_;
  const Matrix<BaseFloat> * scores_; // Scores between all elements
  BaseFloat total_distance_;
};

}

#endif  // KALDI_IVECTOR_GROUP_CLUSTERABLE_H_
