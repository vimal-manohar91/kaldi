// nnet3/nnet-example.cc

// Copyright 2012-2015    Johns Hopkins University (author: Daniel Povey)
//                2014    Vimal Manohar

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

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"
#include "lat/lattice-functions.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet3 {

void NnetIo::Write(std::ostream &os, bool binary) const {
  KALDI_ASSERT(features.NumRows() == static_cast<int32>(indexes.size()));
  WriteToken(os, binary, "<NnetIo>");
  WriteToken(os, binary, name);
  WriteIndexVector(os, binary, indexes);
  features.Write(os, binary);
  WriteToken(os, binary, "<DW>");  // for DerivWeights.  Want to save space.
  WriteVectorAsChar(os, binary, deriv_weights);
  WriteToken(os, binary, "</NnetIo>");
  KALDI_ASSERT(static_cast<size_t>(features.NumRows()) == indexes.size());
}

void NnetIo::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<NnetIo>");
  ReadToken(is, binary, &name);
  ReadIndexVector(is, binary, &indexes);
  features.Read(is, binary);
  std::string token;
  ReadToken(is, binary, &token);
  // in the future this back-compatibility code can be reworked.
  if (token != "</NnetIo>") {
    KALDI_ASSERT(token == "<DW>");
    ReadVectorAsChar(is, binary, &deriv_weights);
    ExpectToken(is, binary, "</NnetIo>");
  }
}

bool NnetIo::operator == (const NnetIo &other) const {
  if (name != other.name) return false;
  if (indexes != other.indexes) return false;
  if (features.NumRows() != other.features.NumRows() ||
      features.NumCols() != other.features.NumCols())
    return false;
  Matrix<BaseFloat> this_mat, other_mat;
  features.GetMatrix(&this_mat);
  other.features.GetMatrix(&other_mat);
  return (ApproxEqual(this_mat, other_mat) &&
          deriv_weights.ApproxEqual(other.deriv_weights));
}

NnetIo::NnetIo(const std::string &name,
               int32 t_begin, const MatrixBase<BaseFloat> &feats,
               int32 skip_frame):
    name(name), features(feats) {
  int32 num_skipped_rows = feats.NumRows();
  KALDI_ASSERT(num_skipped_rows > 0);
  indexes.resize(num_skipped_rows);  // sets all n,t,x to zeros.
  for (int32 i = 0; i < num_skipped_rows; i++)
    indexes[i].t = t_begin + i * skip_frame;
}

NnetIo::NnetIo(const std::string &name,
               const VectorBase<BaseFloat> &deriv_weights,
               int32 t_begin, const MatrixBase<BaseFloat> &feats,
               int32 skip_frame):
    name(name), features(feats), deriv_weights(deriv_weights) {
  int32 num_skipped_rows = feats.NumRows();
  KALDI_ASSERT(num_skipped_rows > 0);
  indexes.resize(num_skipped_rows);  // sets all n,t,x to zeros.
  for (int32 i = 0; i < num_skipped_rows; i++)
    indexes[i].t = t_begin + i * skip_frame;
}

void NnetIo::Swap(NnetIo *other) {
  name.swap(other->name);
  indexes.swap(other->indexes);
  features.Swap(&(other->features));
  deriv_weights.Swap(&(other->deriv_weights));
}

NnetIo::NnetIo(const std::string &name,
               int32 dim,
               int32 t_begin,
               const Posterior &labels,
               int32 skip_frame):
    name(name) {
  int32 num_skipped_rows = labels.size();
  KALDI_ASSERT(num_skipped_rows > 0);
  SparseMatrix<BaseFloat> sparse_feats(dim, labels);
  features = sparse_feats;
  indexes.resize(num_skipped_rows);  // sets all n,t,x to zeros.
  for (int32 i = 0; i < num_skipped_rows; i++)
    indexes[i].t = t_begin + i * skip_frame;
}

NnetIo::NnetIo(const std::string &name,
               const VectorBase<BaseFloat> &deriv_weights,
               int32 dim,
               int32 t_begin,
               const Posterior &labels,
               int32 skip_frame):
    name(name), deriv_weights(deriv_weights) {
  int32 num_skipped_rows = labels.size();
  KALDI_ASSERT(num_skipped_rows > 0);
  SparseMatrix<BaseFloat> sparse_feats(dim, labels);
  features = sparse_feats;
  indexes.resize(num_skipped_rows);  // sets all n,t,x to zeros.
  for (int32 i = 0; i < num_skipped_rows; i++)
    indexes[i].t = t_begin + i * skip_frame;
}

void NnetExample::Write(std::ostream &os, bool binary) const {
#ifdef KALDI_PARANOID
  KALDI_ASSERT(NumOutputs(eg) > 0);
#endif
  // Note: weight, label, input_frames and spk_info are members.  This is a
  // struct.
  WriteToken(os, binary, "<Nnet3Eg>");
  WriteToken(os, binary, "<NumIo>");
  int32 size = io.size();
  KALDI_ASSERT(size > 0 && "Writing empty nnet example");
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    io[i].Write(os, binary);
  WriteToken(os, binary, "</Nnet3Eg>");
}

void NnetExample::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Nnet3Eg>");
  ExpectToken(is, binary, "<NumIo>");
  int32 size;
  ReadBasicType(is, binary, &size);
  if (size <= 0 || size > 1000000)
    KALDI_ERR << "Invalid size " << size;
  io.resize(size);
  for (int32 i = 0; i < size; i++)
    io[i].Read(is, binary);
  ExpectToken(is, binary, "</Nnet3Eg>");
}


void NnetExample::Compress(int32 format) {
  std::vector<NnetIo>::iterator iter = io.begin(), end = io.end();
  // calling features.Compress() will do nothing if they are sparse or already
  // compressed.
  for (; iter != end; ++iter)
    iter->features.Compress(format);
}

} // namespace nnet3
} // namespace kaldi
