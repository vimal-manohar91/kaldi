// segmenter/gaussian-stats.h

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
#include "util/common-utils.h"
#include "itf/clusterable-itf.h"
#include "segmenter/gaussian-stats-clusterable.h"

namespace kaldi {

void GaussianStatsOptions::Register(OptionsItf *opts) {
  opts->Register("use-full-covar", &use_full_covar,
                 "Use full covariance Gaussians.");
  opts->Register("variance-floor", &var_floor,
                 "Floor variances during Gaussian estimation.");
  opts->Register("distance-metric", &distance_metric,
                 "Choose a distance metric among kl2 | glr | bic");
  opts->Register("threshold", &threshold, 
                 "Threshold for merging or splitting segments. "
                 "Also used as the BIC penalty.");
}

void GaussianStatsOptions::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<GaussianStatsOptions>");
  if (use_full_covar) { WriteToken(os, binary, "<FC>"); }
  WriteToken(os, binary, "<VarFloor>");
  WriteBasicType(os, binary, var_floor);
  WriteToken(os, binary, "<Metric>");
  WriteToken(os, binary, distance_metric);
  WriteToken(os, binary, "<Threshold>");
  WriteBasicType(os, binary, threshold);
  WriteToken(os, binary, "</GaussianStatsOptions>");
}

void GaussianStatsOptions::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<GaussianStatsOptions>");
  std::string token;
  while (token != "</GaussianStatsOptions>") {
    ReadToken(is, binary, &token);
    if (token == "<FC>") {
      use_full_covar = true;
    } else if (token == "<VarFloor>") {
      ReadBasicType(is, binary, &var_floor);
    } else if (token == "<Metric>") {
      ReadToken(is, binary, &token);
      if (token == "kl2" || token == "glr" || token == "bic") 
        distance_metric = token;
      else 
        KALDI_ERR << "Unknown distance metric " << token;
    } else if (token == "<Threshold>") {
      ReadBasicType(is, binary, &threshold);
    } else if (token == "</GaussianStatsOptions>") {
      break;
    } else {
      KALDI_ERR << "Unknown token " << token;
    }
  }
}

GaussianStatsClusterable::GaussianStatsClusterable(
    int32 dim, const GaussianStatsOptions &opts)
      : count_(0), opts_(opts) {
  Resize(dim);
}

void GaussianStatsClusterable::Resize(int32 dim) {
  x_.Resize(dim);

  if (IsFullCovariance())
    x2_.Resize(dim);
  else
    x2_diag_.Resize(dim);
}

void GaussianStatsClusterable::AddStats(const VectorBase<BaseFloat> &vec, 
                                        BaseFloat weight) {
  count_ += weight;
  x_.AddVec(weight, vec);
  if (IsFullCovariance())
    x2_.AddVec2(weight, vec);
  else {
    x2_diag_.AddVecVec(weight, vec, vec, 1.0);
  }
}

void GaussianStatsClusterable::Add(const Clusterable &other_in) {
  KALDI_ASSERT(other_in.Type() == "gaussian-stats");
  const GaussianStatsClusterable *other = 
    static_cast<const GaussianStatsClusterable*>(&other_in);

  count_ += other->count_;
  x_.AddVec(1.0, other->x_);
  KALDI_ASSERT(IsFullCovariance() == other->IsFullCovariance());

  if (IsFullCovariance()) {
    x2_.AddSp(1.0, other->x2_);
  } else {
    x2_diag_.AddVec(1.0, other->x2_diag_);
  }
}

void GaussianStatsClusterable::Sub(const Clusterable &other_in) {
  KALDI_ASSERT(other_in.Type() == "gaussian-stats");
  const GaussianStatsClusterable *other = 
    static_cast<const GaussianStatsClusterable*>(&other_in);

  count_ -= other->count_;
  x_.AddVec(-1.0, other->x_);
  KALDI_ASSERT(IsFullCovariance() == other->IsFullCovariance());

  if (IsFullCovariance()) {
    x2_.AddSp(-1.0, other->x2_);
  } else {
    x2_diag_.AddVec(-1.0, other->x2_diag_);
  }
}

void GaussianStatsClusterable::SetZero() {
  count_ = 0.0;
  x_.SetZero(); 
  x2_.SetZero(); 
  x2_diag_.SetZero();
}

Clusterable *GaussianStatsClusterable::Copy() const {
  KALDI_ASSERT(Dim() > 0);
  GaussianStatsClusterable *ans = new GaussianStatsClusterable(Dim(), opts_);
  ans->Add(*this);
  return ans;
}

void GaussianStatsClusterable::Scale(BaseFloat f) {
  KALDI_ASSERT(f >= 0);
  count_ *= f;
  x_.Scale(f);
  x2_.Scale(f);
  x2_diag_.Scale(f);
}

void GaussianStatsClusterable::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "GaussStatsCL");
  WriteBasicType(os, binary, count_);
  x_.Write(os, binary);
  opts_.Write(os, binary);
  if (IsFullCovariance())
    x2_.Write(os, binary);
  else
    x2_diag_.Write(os, binary);
}

Clusterable* GaussianStatsClusterable::ReadNew(
    std::istream &is, bool binary) const {
  GaussianStatsClusterable *gc = new GaussianStatsClusterable();
  gc->Read(is, binary);
  return gc;
}

void GaussianStatsClusterable::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "GaussianStatsCL");  // magic string.
  ReadBasicType(is, binary, &count_);
  x_.Read(is, binary);
  opts_.Read(is, binary);
  if (IsFullCovariance())
    x2_.Read(is, binary);
  else 
    x2_diag_.Read(is, binary);
}

void GaussianStatsClusterable::GetMean(Vector<BaseFloat> *mean) const {
  KALDI_ASSERT(mean->Dim() == Dim());
  mean->CopyFromVec(x_);
  mean->Scale(1.0 / count_);
}

void GaussianStatsClusterable::GetDiagVars(Vector<BaseFloat> *vars, 
                                           Vector<BaseFloat> *mean) const {
  KALDI_ASSERT(vars && vars->Dim() == Dim());
  KALDI_ASSERT(!mean || mean->Dim() == Dim());

  if (IsFullCovariance())
    vars->CopyDiagFromSp(x2_);
  else
    vars->CopyFromVec(x2_diag_);
  vars->Scale(1.0 / count_);

  if (mean) {
    GetMean(mean);
    vars->AddVec2(-1.0, *mean);
  } else {
    mean = new Vector<BaseFloat>(Dim());
    GetMean(mean);
    vars->AddVec2(-1.0, *mean);
    delete mean;
    mean = NULL;
  }
}

void GaussianStatsClusterable::GetFullCovariance(
    SpMatrix<BaseFloat> *covar, Vector<BaseFloat> *mean) const {
  KALDI_ASSERT(covar && covar->NumCols() == Dim());
  KALDI_ASSERT(!mean || mean->Dim() == Dim());

  if (IsFullCovariance())
    covar->CopyFromSp(x2_);
  else {
    covar->SetZero();
    covar->AddDiagVec(1.0, x2_diag_);
  }
  covar->Scale(1.0 / count_);

  if (mean) {
    GetMean(mean);
    covar->AddVec2(-1.0, *mean);
  } else {
    mean = new Vector<BaseFloat>(Dim());
    GetMean(mean);
    covar->AddVec2(-1.0, *mean);
    delete mean;
    mean = NULL;
  }
}

BaseFloat GaussianStatsClusterable::Objf() const {
  if (opts_.distance_metric == "glr") {
    return ObjfGLR(*this, opts_.var_floor);
  } 
  return -std::numeric_limits<BaseFloat>::max();
}

BaseFloat GaussianStatsClusterable::Distance(
    const Clusterable &other_in) const {
  KALDI_ASSERT(other_in.Type() == "gaussian-stats");

  const GaussianStatsClusterable *other = 
    static_cast<const GaussianStatsClusterable*> (&other_in);

  if (opts_.distance_metric == "kl2")
    return DistanceKL2Diag(*this, *other, opts_.var_floor);
  else if (opts_.distance_metric == "glr") 
    return DistanceGLR(*this, *other, opts_.var_floor);
  else if (opts_.distance_metric == "bic")
    return DistanceBIC(*this, *other, opts_.threshold, opts_.var_floor);
  else 
    KALDI_ERR << "Unkown distance metric " << opts_.distance_metric;

  return -std::numeric_limits<BaseFloat>::max();
}

BaseFloat ObjfGLR(const GaussianStatsClusterable &stats,
                  BaseFloat var_floor) {
  double ans = -0.5 * stats.Normalizer() * stats.Dim() 
    - 0.5 * stats.Normalizer() * stats.Dim() * M_LOG_2PI;

  if (stats.IsFullCovariance()) {
    SpMatrix<BaseFloat> covar(stats.Dim());
    stats.GetFullCovariance(&covar);
    covar.ApplyFloor(var_floor);
    ans -= 0.5 * stats.Normalizer() * covar.LogDet();
  } else {
    Vector<BaseFloat> vars(stats.Dim());
    stats.GetDiagVars(&vars);
    vars.ApplyFloor(var_floor);
    ans -= 0.5 * stats.Normalizer() * vars.SumLog();
  }

  return ans;
}

void EstGaussian(const MatrixBase<BaseFloat> &feats, BaseFloat scale,
                 GaussianStatsClusterable *gauss) {
  KALDI_ASSERT(gauss);
  gauss->SetZero();
  for (int32 i = 0; i < feats.NumRows(); i++) {
    gauss->AddStats(feats.Row(i), scale);
  }
}

BaseFloat DistanceKL2Diag(const GaussianStatsClusterable &stats1,
                          const GaussianStatsClusterable &stats2,
                          BaseFloat var_floor) {
  KALDI_ASSERT(stats1.Dim() == stats2.Dim());
  Vector<BaseFloat> mean1(stats1.Dim());
  Vector<BaseFloat> vars1(stats1.Dim());
  stats1.GetDiagVars(&vars1, &mean1);
  vars1.ApplyFloor(var_floor);

  Vector<BaseFloat> mean2(stats2.Dim());
  Vector<BaseFloat> vars2(stats2.Dim());
  stats2.GetDiagVars(&vars2, &mean2);
  vars2.ApplyFloor(var_floor);
    
  double ans = -stats1.Dim();
  for (int32 i = 0; i < stats1.Dim(); i++) {
    ans += 0.5 * (vars1(i) / vars2(i) + vars2(i) / vars1(i) 
        + (mean2(i) - mean1(i)) * (mean2(i) - mean1(i)) 
        * (1.0 / vars1(i) + 1.0 / vars2(i))); 
  }

  return ans;
}

BaseFloat DistanceGLR(const GaussianStatsClusterable &stats1,
                      const GaussianStatsClusterable &stats2,
                      BaseFloat var_floor) {
  GaussianStatsClusterable *combined_stats = 
    static_cast<GaussianStatsClusterable*>(stats1.Copy());
  combined_stats->Add(stats2);

  double ans = 
    ObjfGLR(stats1, var_floor) + ObjfGLR(stats2, var_floor) 
    - ObjfGLR(*combined_stats, var_floor);
  delete combined_stats;
  return ans;
}

BaseFloat DistanceBIC(const GaussianStatsClusterable &stats1,
                      const GaussianStatsClusterable &stats2,
                      BaseFloat penalty,
                      BaseFloat var_floor) {
  KALDI_ASSERT(stats1.Dim() == stats2.Dim());
  int32 D = stats1.Dim();
  double ans = DistanceGLR(stats1, stats2, var_floor);

  SpMatrix<BaseFloat> covar1(D);
  stats1.GetFullCovariance(&covar1);
  SpMatrix<BaseFloat> covar2(D);
  stats2.GetFullCovariance(&covar2);
  GaussianStatsClusterable *combined_stats = 
    static_cast<GaussianStatsClusterable*>(stats1.Copy());
  combined_stats->Add(stats2);
  SpMatrix<BaseFloat> combined_covar(D);
  combined_stats->GetFullCovariance(&combined_covar);
  covar1.ApplyFloor(var_floor);
  covar2.ApplyFloor(var_floor);
  combined_covar.ApplyFloor(var_floor);
  double ans2 = 0.5 * (combined_stats->Normalizer() * combined_covar.LogDet() 
      - stats1.Normalizer() * covar1.LogDet() 
      - stats2.Normalizer() * covar2.LogDet());
  if (!kaldi::ApproxEqual(ans, ans2)) 
    KALDI_LOG << "ans = " << ans << " vs ans2 = " << ans2;

  delete combined_stats;
  double P = 0.5 * (D + (stats1.IsFullCovariance() ? (D*(D+1))/2 : D))
    * Log(stats1.Normalizer() + stats2.Normalizer());
  ans -= penalty * P;
  return ans;
}

}   // end namespace kaldi
