// nnet3/nnet-supervision-example.h

// Copyright 2012-2015  Johns Hopkins University (author: Daniel Povey)
//                2014  Vimal Manohar
//                2015  Pegah Ghahremani 

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

#ifndef KALDI_NNET3_NNET_SUPERVISION_EXAMPLE_H_
#define KALDI_NNET3_NNET_SUPERVISION_EXAMPLE_H_

#include "nnet3/nnet-nnet.h"
#include "hmm/posterior.h"
#include "util/table-types.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet3 {
// This is a base class for the output of examples in nnet3, which is used to store output. 
struct NnetSupervision {
  std::string name;
  /// "indexes" is a vector the same length as features.NumRows(), explaining
  /// the meaning of each row of the "features" matrix.  Note: the "n" values
  /// in the indexes will always be zero in individual examples, but in general
  /// nonzero after we aggregate the examples into the minibatch level.
  std::vector<Index> indexes;
  
  NnetSupervision() { }; 

  NnetSupervision(std::string name, std::vector<Index> indexes):
    name(name), indexes(indexes) { }
  
  NnetSupervision(std::string name): name(name) { }

  virtual ~NnetSupervision() { };
   
  /// Use default copy constructor and assignment operators.
  virtual void Write(std::ostream &os, bool binary) const;    

  virtual void Read(std::istream &is, bool binary)  { };
  
  /// Returns a string such as "NnetIo", describing the type of supervision.
  virtual std::string Type() { return "NnetSupervision"; };

  /// Returns a new supervision of the given type e.g. "NnetChainSupervision",
  /// or NULL if no such supervision type exists.
  static NnetSupervision* NewSupervisionOfType(const std::string &sup_type);

  protected:
   friend struct NnetIo;
   friend struct NnetChainSupervision;

};

} // namespace nnet3 
} // namespace kaldi
#endif // KALDI_NNET3_NNET_SUPERVISION_EXAMPLE_H_  
