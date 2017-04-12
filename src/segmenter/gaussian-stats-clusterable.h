// segmenter/gaussian-stats-clusterable.h

// Copyright 2017   Vimal Manohar (Johns Hopkins University)

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

#include "base/kaldi-common.h"
#include "itf/clusterable-itf.h"
#include "util/common-utils.h"

#ifndef KALDI_SEGMENTER_GAUSSIAN_STATS_H_
#define KALDI_SEGMENTER_GAUSSIAN_STATS_H_

namespace kaldi {

struct GaussianStatsOptions {
  bool use_full_covar;
  BaseFloat var_floor;
  std::string distance_metric;
  BaseFloat threshold;

  GaussianStatsOptions(): 
    use_full_covar(true), var_floor(0.01), distance_metric("glr"),
    threshold(0) { }

  void Register(OptionsItf *opts);

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};

class GaussianStatsClusterable: public Clusterable {
 public:
  GaussianStatsClusterable() : count_(0) { }
  GaussianStatsClusterable(int32 dim, const GaussianStatsOptions &opts);

  void Resize(int32 dim);
  void AddStats(const VectorBase<BaseFloat> &vec, BaseFloat weight);

  virtual void SetZero();
  virtual void Add(const Clusterable &other);
  virtual void Sub(const Clusterable &other);
  virtual BaseFloat Normalizer() const { return count_; }
  virtual Clusterable *Copy() const;
  virtual void Scale(BaseFloat f);
  virtual std::string Type() const { return "gaussian-stats"; }
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Clusterable* ReadNew(std::istream &is, bool binary) const;
  virtual ~GaussianStatsClusterable() { }

  virtual BaseFloat Objf() const;
  BaseFloat Distance(const Clusterable &other_in) const;
  
  void GetMean(Vector<BaseFloat> *mean) const;
  void GetDiagVars(Vector<BaseFloat> *vars, 
                   Vector<BaseFloat> *mean = NULL) const;
  void GetFullCovariance(SpMatrix<BaseFloat> *covar,
                         Vector<BaseFloat> *mean = NULL) const;

  int32 Dim() const { return x_.Dim(); }
  bool IsFullCovariance() const { return opts_.use_full_covar; }

 private:
  Vector<BaseFloat> x_;
  SpMatrix<BaseFloat> x2_;
  Vector<BaseFloat> x2_diag_;
  BaseFloat count_;

  GaussianStatsOptions opts_;

  void Read(std::istream &is, bool binary);
};

void EstGaussian(const MatrixBase<BaseFloat> &feats, BaseFloat scale,
                 GaussianStatsClusterable *gauss);

BaseFloat ObjfGLR(const GaussianStatsClusterable &stats,
                  BaseFloat var_floor);

BaseFloat DistanceKL2Diag(const GaussianStatsClusterable &stats1,
                          const GaussianStatsClusterable &stats2,
                          BaseFloat var_floor);

/**
 * Computes the log-likelihood ratio between the features corresponding to the 
 * statistics in stats1 and stats2 distributed according to two independent
 * Gaussians versus a single Gaussian. 
 * All the Gaussian paramters are assumed to 
 * be Maximum likelihood estimates w.r.t. the corresponding stats.
 * 
 * dist = - 0.5 * n_1 * log det(Sigma_1) - 0.5 * n_2 * log det(Sigma_2) 
 *        + 0.5 * (n_1 + n_2) * log det(Sigma) 
 **/ 
BaseFloat DistanceGLR(const GaussianStatsClusterable &stats1,
                      const GaussianStatsClusterable &stats2,
                      BaseFloat var_floor);

/**
 * Computes the Bayesian-information criterion distance between the features
 * corresponding to the statistics in stats1 and stats2 distributed according to
 * two independent Gaussians versus a single Gaussian. 
 * All the Gaussian paramters are assumed to 
 * be Maximum likelihood estimates w.r.t. the corresponding stats.
 * 
 * dist = - 0.5 * n_1 * log det(Sigma_1) - 0.5 * n_2 * log det(Sigma_2) 
 *        + 0.5 * (n_1 + n_2) * log det(Sigma) - \lambda * P,
 * where P = 0.5 * (D + (D * (D+1)) / 2) + log(n_1 + n_2) 
 **/ 
BaseFloat DistanceBIC(const GaussianStatsClusterable &stats1,
                      const GaussianStatsClusterable &stats2,
                      BaseFloat penalty,  // lambda
                      BaseFloat var_floor);

}  // end namespace kaldi

#endif  // KALDI_SEGMENTER_GAUSSIAN_STATS_H_
