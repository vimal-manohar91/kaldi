// ivector/plda-clusterable.cc

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

#include "base/kaldi-math.h"
#include "itf/clusterable-itf.h"
#include "itf/options-itf.h"
#include "segmenter/plda-clusterable.h"
#include "ivector/plda.h"

namespace kaldi {

void PldaClusterableOptions::Register(OptionsItf *opts) {
  opts->Register("use-buggy-eqn", &use_buggy_eqn,
                 "Use buggy equation (6) in original PLDA paper.");
  opts->Register("use-nfactor", &use_nfactor,
                 "Add a 0.5 * log(N) penalty term to the objective.");
  opts->Register("normalize-distance", &normalize_distance,
                 "Normalize PLDA distance when considereing clusters to merge.");
  opts->Register("normalize-loglike", &normalize_loglike,
                 "Normalize PLDA log-likelihood by weight in the cluster.");
  opts->Register("apply-logistic", &apply_logistic,
                 "Apply logistic transformation on the PLDA distance.");
  opts->Register("use-avg-ivector", &use_avg_ivector,
                 "Use average ivector");
  opts->Register("bic-penalty", &bic_penalty,
                 "BIC penalty");
}

void PldaClusterable::AddStats(const VectorBase<BaseFloat> &vec,
                               BaseFloat weight) {
  KALDI_ASSERT(vec.Dim() == plda_->Dim());
  weight_ += weight;
  stats_.AddVec(weight, vec);
  sumsq_ += weight * VecVec(vec, vec);
}

void PldaClusterable::Add(const Clusterable &other_in) {
  KALDI_ASSERT(other_in.Type() == "plda");
  const PldaClusterable *other =
      static_cast<const PldaClusterable*>(&other_in);

  weight_ += other->weight_;
  stats_.AddVec(1.0, other->stats_);
  sumsq_ += other->sumsq_;
  
  points_.insert(other->points_.begin(), other->points_.end());
}

void PldaClusterable::Sub(const Clusterable &other_in) {
  KALDI_ASSERT(other_in.Type() == "plda");
  const PldaClusterable *other =
      static_cast<const PldaClusterable*>(&other_in);
  weight_ -= other->weight_;
  sumsq_ -= other->sumsq_;
  stats_.AddVec(-1.0, other->stats_);
  if (weight_ < 0.0) {
    if (weight_ < -0.1 && weight_ < -0.0001 * fabs(other->weight_)) {
      // a negative weight may indicate an algorithmic error if it is
      // encountered.
      KALDI_WARN << "Negative weight encountered " << weight_;
    }
    weight_ = 0.0;
  }
  if (weight_ == 0.0) {
    sumsq_ = 0.0;
    stats_.Set(0.0);
  }

  for (std::set<int32>::const_iterator itr_i = other->points_.begin();
       itr_i != other->points_.end(); ++itr_i) {
    points_.erase(*itr_i);
  }
}

Clusterable* PldaClusterable::Copy() const {
  PldaClusterable *ans = new PldaClusterable(opts_, plda_);
  ans->Add(*this);
  return ans;
}

void PldaClusterable::Scale(BaseFloat f) {
  KALDI_ASSERT(f >= 0.0);
  weight_ *= f;
  stats_.Scale(f);
  sumsq_ *= f;
}

void PldaClusterable::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "PldaCL");  // magic string.
  std::vector<int32> vec(points_.begin(), points_.end());
  WriteIntegerVector(os, binary, vec);
  WriteToken(os, binary, "<Weight>");
  WriteBasicType(os, binary, weight_);
  WriteToken(os, binary, "<Sumsq>");  
  WriteBasicType(os, binary, sumsq_);
  WriteToken(os, binary, "<Stats>");    
  stats_.Write(os, binary);
}

Clusterable* PldaClusterable::ReadNew(std::istream &is, bool binary) const {
  PldaClusterable *pc = new PldaClusterable(opts_, plda_);
  pc->Read(is, binary);
  return pc;
}

void PldaClusterable::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "PldaCL");  // magic string.
  std::vector<int32> vec;
  ReadIntegerVector(is, binary, &vec);
  points_.clear();
  std::copy(vec.begin(), vec.end(), std::inserter(points_, points_.end()));
  ExpectToken(is, binary, "<Weight>");
  ReadBasicType(is, binary, &weight_);
  ExpectToken(is, binary, "<Sumsq>");  
  ReadBasicType(is, binary, &sumsq_);
  ExpectToken(is, binary, "<Stats>");    
  stats_.Read(is, binary);
}

PldaClusterable::PldaClusterable(const PldaClusterableOptions &opts,
                                 const Plda *plda,
                                 const std::set<int32> &points,
                                 const Vector<BaseFloat> &vector,
                                 BaseFloat weight) :
    opts_(opts), plda_(plda), points_(points), weight_(weight), stats_(vector),
    sumsq_(0.0) {
  stats_.Scale(weight);
  KALDI_ASSERT(weight >= 0.0);
  sumsq_ = VecVec(vector, vector) * weight;
}

PldaClusterable::PldaClusterable(const Plda *plda,
                                 const std::set<int32> &points,
                                 const Vector<BaseFloat> &vector,
                                 BaseFloat weight) :
    plda_(plda), points_(points), weight_(weight), stats_(vector),
    sumsq_(0.0) {
  stats_.Scale(weight);
  KALDI_ASSERT(weight >= 0.0);
  sumsq_ = VecVec(vector, vector) * weight;
}

BaseFloat PldaClusterable::Objf() const {
  int32 dim = plda_->Dim();
  const Vector<double> &psi = plda_->BetweenClassCovariance();
  
  KALDI_ASSERT(dim == stats_.Dim());

  if (opts_.use_avg_ivector) {
    // Approximate log-likelihood using the averaged i-vector
    double logdet = 0.0;
    double between_class_term = 0.0;
    for (size_t t = 0; t < dim; t++) {
      between_class_term -= stats_(t) * stats_(t) / weight_ / weight_ 
        * psi(t) / (1.0 + psi(t));
      logdet += Log(1.0 + psi(t));
    }
    double sumsq = VecVec(stats_, stats_) / weight_ / weight_;

    double ans = (-0.5 * dim * M_LOG_2PI  
      - 0.5 * logdet - 0.5 * sumsq - 0.5 * between_class_term);
    if (opts_.normalize_loglike)
      ans /= weight_;
    return ans;
  }

  double direct_sumsq;
  if (weight_ > std::numeric_limits<BaseFloat>::min()) {
    direct_sumsq = VecVec(stats_, stats_) / weight_;
  } else {
    direct_sumsq = 0.0;
  }
  // ans is a negated weighted sum of squared distances; it should not be
  // positive.
  double ans = -(sumsq_ - direct_sumsq); 
  
  for (size_t t = 0; t < dim; t++) {
    ans -= stats_(t) * stats_(t) 
           / (weight_ * weight_ * psi(t) + weight_);  // First term inside \exp in eqn(6) in original paper
    ans -= Log(1.0 + weight_ * psi(t));   // Logdet term in eqn(6) in original paper
    if (opts_.use_buggy_eqn)     // Use as is in eqn(6) in the original paper
      ans += Log(weight_); 
    else if (opts_.use_nfactor)  // Encourage clusters of equal size.
      ans -= Log(weight_); 
    // else Use eqn(9) in the paper
    // http://cs.joensuu.fi/pages/tkinnu/webpage/pdf/Practical_PLDA_DSP2014.pdf
  }
  ans -= weight_ * dim * M_LOG_2PI;
  ans *= 0.5;

  // ans is now the log-likelihood as in eqn(6) or a modified version of it.
  if (opts_.use_buggy_eqn || opts_.use_nfactor) {
    if (opts_.normalize_loglike)
      ans /= weight_;
    return ans;
  }

  // Test code: 
  // Verify expression by comparing with eqn(9) from  
  // http://cs.joensuu.fi/pages/tkinnu/webpage/pdf/Practical_PLDA_DSP2014.pdf
  double logdet = 0.0;
  double between_class_term = 0.0;
  for (size_t t = 0; t < dim; t++) {
    between_class_term -= stats_(t) * stats_(t) * psi(t) / (1.0 + weight_ * psi(t));
    logdet += Log(1.0 + weight_ * psi(t));
  }

  double ans2 = -0.5 * weight_ * dim * M_LOG_2PI  
    - 0.5 * logdet - 0.5 * sumsq_ - 0.5 * between_class_term;
  
  KALDI_ASSERT(kaldi::ApproxEqual(ans2, ans));

  if (opts_.normalize_loglike)
    ans /= weight_;
  return ans;     // Here ans2 ~ ans
}

double Sigmoid(double x) {
  if (x > 0.0) {
    x = 1.0 / (1.0 + Exp(-x));
  } else {
    double ex = Exp(x);
    x = ex / (ex + 1.0);
  }
  return x;
}

BaseFloat PldaClusterable::Distance(const Clusterable &other_in) const {
  const PldaClusterable *other =
      static_cast<const PldaClusterable*>(&other_in);
  PldaClusterable *copy = static_cast<PldaClusterable*>(this->Copy());
  copy->Add(*other);
  double llr = copy->Objf() - other->Objf() - this->Objf();
 
  if (opts_.use_avg_ivector || 
      (this->weight_ == 1 && other->weight_ == 1 && std::abs(llr) > 0.01)) {
    // Use the average vector in a cluster as proxy for representing the 
    // cluster. 
    Vector<double> vec1(this->stats_);
    vec1.Scale(1.0 / this->weight_);
    Vector<double> vec2(other->stats_);
    vec2.Scale(1.0 / other->weight_);

    double score = plda_->LogLikelihoodRatio(vec1, this->weight_, vec2);
    if (opts_.use_avg_ivector) {
      llr = score;
    }

    // Test the log-likelihood ratio matches the standard function in Plda class.
    // KALDI_ASSERT(kaldi::ApproxEqual(llr, score));  
  }

  if (opts_.normalize_distance)
    // Normalize the log-likelihood ratio by the weight.
    llr /= copy->weight_;
  
  if (opts_.bic_penalty != 0.0) {
    llr += opts_.bic_penalty * 2 * plda_->Dim() * Log(weight_);
  }

  delete copy;

  if (opts_.apply_logistic)
    return Sigmoid(-llr);   // apply sigmoid transformation to distance.
  else
    return -llr;    // convert score to distance
}

}  // end namespace kaldi
