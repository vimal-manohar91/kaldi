// ivector/ivector-clusterable.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University

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

#ifndef KALDI_IVECTOR_IVECTOR_CLUSTERABLE_H_
#define KALDI_IVECTOR_IVECTOR_CLUSTERABLE_H_

#include <algorithm>
#include <string>
#include "base/kaldi-math.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "itf/clusterable-itf.h"
#include "tree/clusterable-classes.h"

namespace kaldi {

class IvectorClusterable: public Clusterable {
 public:
  IvectorClusterable(): weight_(0.0) {}

  IvectorClusterable(const Vector<BaseFloat> &vector,
                     BaseFloat weight);

  virtual std::string Type() const {  return "ivector"; }
  // Objf is negated weighted sum of squared distances.
  virtual BaseFloat Objf() const;
  virtual void SetZero() { weight_ = 0.0; stats_.Set(0.0); }
  virtual void Add(const Clusterable &other_in);
  virtual void Sub(const Clusterable &other_in);
  virtual BaseFloat Normalizer() const { return weight_; }
  virtual Clusterable *Copy() const;
  virtual void Scale(BaseFloat f);
  virtual BaseFloat Distance(const Clusterable &other) const;
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Clusterable *ReadNew(std::istream &is, bool binary) const;
  virtual ~IvectorClusterable() {}

 protected:
  double weight_;  // sum of weights of the source vectors.  Never negative.
  Vector<double> stats_; // Equals the weighted sum of the source vectors.
                  
 private:
  void Read(std::istream &is, bool binary);
};

}

#endif  // KALDI_IVECTOR_IVECTOR_CLUSTERABLE_H_
