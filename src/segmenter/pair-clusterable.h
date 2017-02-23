// segmenter/pair-clusterable.h

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

#ifndef KALDI_SEGMENTER_PAIR_CLUSTERABLE_H_
#define KALDI_SEGMENTER_PAIR_CLUSTERABLE_H_

#include <vector>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "itf/clusterable-itf.h"

namespace kaldi {

class PairClusterable: public Clusterable {
 public:
  PairClusterable(): clusterable1_(NULL), clusterable2_(NULL), 
                     weight1_(1.0), weight2_(1.0) { }
  PairClusterable(Clusterable* c1, Clusterable* c2,
                  BaseFloat weight1, BaseFloat weight2):
    clusterable1_(c1), clusterable2_(c2), 
    weight1_(weight1), weight2_(weight2) { }
  virtual std::string Type() const { return "pair"; }
  virtual BaseFloat Objf() const;
  virtual void SetZero();
  virtual void Add(const Clusterable &other_in);
  virtual void Sub(const Clusterable &other_in);
  virtual BaseFloat Normalizer() const {
    return clusterable1_->Normalizer();
  }
  virtual Clusterable *Copy() const;
  virtual void Scale(BaseFloat f);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Clusterable *ReadNew(std::istream &is, bool binary) const;
  virtual BaseFloat Distance(const Clusterable &other_in) const;
  virtual BaseFloat MergeThreshold(const Clusterable &other_in) const;

  virtual ~PairClusterable() {}

  virtual Clusterable* clusterable1() const { return clusterable1_; }
  virtual Clusterable* clusterable2() const { return clusterable2_; }
  virtual BaseFloat Weight1() const { return weight1_; }
  virtual BaseFloat Weight2() const { return weight2_; }

 private:
  Clusterable *clusterable1_;
  Clusterable *clusterable2_;
  BaseFloat weight1_;
  BaseFloat weight2_;
};

}  // end namespace kaldi

#endif  // KALDI_SEGMENTER_PAIR_CLUSTERABLE_H_
