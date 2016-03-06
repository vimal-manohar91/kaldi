// base/kaldi-types-extra.cc

// Copyright 2014  Vimal Manohar

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
#include "base/kaldi-types-extra.h"
#include "base/kaldi-types.h"

namespace kaldi {

template<typename Real>
void SignedLogReal<Real>::SetZero() {
  sign_ = false;
  log_f_ = kLogZeroDouble;
}

template<typename Real>
void SignedLogReal<Real>::SetOne() {
  sign_ = false;
  log_f_ = 0.0;
}

template<typename Real>
void SignedLogReal<Real>::Set(Real f) {
  if (f < 0.0) {
    sign_ = true;
    log_f_ = static_cast<Real>(kaldi::Log(static_cast<double>(-f)));
  } else {
    sign_ = false;
    log_f_ = static_cast<Real>(kaldi::Log(static_cast<double>(f)));
  }
}

template<typename Real>
void SignedLogReal<Real>::SetRandn() {
  Set(kaldi::RandGauss());
}

template<typename Real>
void SignedLogReal<Real>::SetRandUniform() {
  Set(kaldi::RandUniform());
}

template<typename Real>
void SignedLogReal<Real>::Log() {
  KALDI_ASSERT(Positive());
  log_f_ = kaldi::Log(log_f_);
}

template<typename Real>
bool SignedLogReal<Real>::IsZero(Real cutoff) const {
  return (log_f_ < kaldi::Log(cutoff)); 
}

template<typename Real>
bool SignedLogReal<Real>::IsOne(Real cutoff) const {
  return ( Positive() && 
      (log_f_ > 0 ? LogSub(log_f_, 0) : LogSub(0, log_f_)) < kaldi::Log(cutoff) );
}

template<typename Real>
bool SignedLogReal<Real>::ApproxEqual(const SignedLogReal<Real> &other, 
                                      float tol) const {
  KALDI_ASSERT(tol > 0);

  if (Sign() == other.sign_) {
    double tmp1 = log_f_;
    double tmp2 = other.LogMagnitude();
    if (tmp1 >= tmp2) {
      return (LogSub(tmp1, tmp2) <= kaldi::Log(tol) + tmp1);
    } else {
      return (LogSub(tmp2, tmp1) <= kaldi::Log(tol) + tmp1);
    }
  } 

  return (LogAdd(log_f_, other.LogMagnitude() <= kaldi::Log(tol) + log_f_));
}

template<typename Real>
bool SignedLogReal<Real>::Equal(const SignedLogReal<Real> &other) const {
  return (sign_ == other.sign_ && log_f_ == other.log_f_);
}

template<typename Real>
template<typename OtherReal>
void SignedLogReal<Real>::Add(const SignedLogReal<OtherReal> &a) {
  if (sign_ == a.Sign()) {
    log_f_ = LogAdd(static_cast<double>(log_f_), 
                    static_cast<double>(a.LogMagnitude()));
  } else {
    if (log_f_ < a.LogMagnitude()) {
      sign_ = !sign_;
      log_f_ = LogSub(static_cast<double>(a.LogMagnitude()), 
                      static_cast<double>(log_f_));
    } else {
      log_f_ = LogSub(static_cast<double>(log_f_), 
                      static_cast<double>(a.LogMagnitude()));
    }
  }
}

template<typename Real>
template<typename OtherReal>
void SignedLogReal<Real>::AddReal(OtherReal f) {
  SignedLogReal<OtherReal> temp(f);
  Add(temp);
}
    
template<typename Real>
template<typename OtherReal>
void SignedLogReal<Real>::AddLogReal(OtherReal log_f) {
  SignedLogReal<OtherReal> temp(false, log_f);
  Add(temp);
}

template<typename Real>
template<typename OtherReal>
void SignedLogReal<Real>::AddMultiplyLogReal(const SignedLogReal<OtherReal> &a, 
    OtherReal log_b) {
  SignedLogReal<OtherReal> temp(false, log_b);
  temp.Multiply(a);
  Add(temp);
}

template<typename Real>
template<typename OtherReal>
void SignedLogReal<Real>::Sub(const SignedLogReal<OtherReal> &a) {
  if (sign_ == a.Sign()) {
    if (log_f_ < a.LogMagnitude()) {
      sign_ = !sign_;
      log_f_ = LogSub(a.LogMagnitude(), log_f_);
    } else {
      log_f_ = LogSub(log_f_, a.LogMagnitude());
    }
  } else {
    log_f_ = LogAdd(log_f_, a.LogMagnitude());
  }
}

template<typename Real>
template<typename OtherReal>
void SignedLogReal<Real>::SubMultiplyLogReal(const SignedLogReal<OtherReal> &a, 
    OtherReal log_b) {
  SignedLogReal<OtherReal> temp(false, log_b);
  temp.Multiply(a);
  Sub(temp);
}

template<typename Real>
template<typename OtherReal>
void SignedLogReal<Real>::Multiply(const SignedLogReal<OtherReal> &a) {
  if (sign_ != a.Sign()) { sign_ = true; }
  else { sign_ = false; }

  log_f_ += a.LogMagnitude();
}

template<typename Real>
template<typename OtherReal>
void SignedLogReal<Real>::MultiplyReal(OtherReal f) {
  SignedLogReal<OtherReal> temp(f);
  Multiply(temp);
}

template<typename Real>
template<typename OtherReal>
void SignedLogReal<Real>::MultiplyLogReal(OtherReal log_f) {
  log_f_ += log_f;
}

template<typename Real>
template<typename OtherReal>
void SignedLogReal<Real>::DivideBy(const SignedLogReal<OtherReal> &a) {
  if (sign_ != a.Sign()) { sign_ = true; }
  else { sign_ = false; }

  log_f_ -= a.LogMagnitude();
}
    
template<typename Real>
SignedLogReal<Real> SignedLogReal<Real>::operator+(
    const SignedLogReal<Real> &a) const {
  SignedLogReal<Real> tmp(*this);
  tmp.Add(a);
  return tmp;
}

template<typename Real>
SignedLogReal<Real> SignedLogReal<Real>::operator*(
    const SignedLogReal<Real> &a) const {
  SignedLogReal<Real> tmp(*this);
  tmp.Multiply(a);
  return tmp;
}

template<typename Real>
SignedLogReal<Real> SignedLogReal<Real>::operator/(
    const SignedLogReal<Real> &a) const {
  SignedLogReal<Real> tmp(*this);
  tmp.DivideBy(a);
  return tmp;
}

template<typename Real>
SignedLogReal<Real> SignedLogReal<Real>::operator-(
    const SignedLogReal<Real> &a) const {
  SignedLogReal<Real> tmp(*this);
  tmp.Sub(a);
  return tmp;
}

template<typename Real>
SignedLogReal<Real> operator-(const SignedLogReal<Real> &a) {
  SignedLogReal<Real> tmp(a);
  tmp.Negate();
  return tmp;
}

template void SignedLogReal<double>::Add(const SignedLogReal<double> &a);
template void SignedLogReal<float>::Add(const SignedLogReal<float> &);
template void SignedLogReal<double>::AddReal(double f);
template void SignedLogReal<float>::AddReal(float f);
template void SignedLogReal<double>::AddLogReal(double f);
template void SignedLogReal<float>::AddLogReal(float f);
template void SignedLogReal<double>::AddMultiplyLogReal(const SignedLogReal<double> &a, double log_b);
template void SignedLogReal<float>::AddMultiplyLogReal(const SignedLogReal<float> &a, float log_b);
template void SignedLogReal<double>::Sub(const SignedLogReal<double> &a);
template void SignedLogReal<float>::Sub(const SignedLogReal<float> &);
template void SignedLogReal<double>::SubMultiplyLogReal(const SignedLogReal<double> &a, double log_b);
template void SignedLogReal<float>::SubMultiplyLogReal(const SignedLogReal<float> &a, float log_b);
template void SignedLogReal<double>::Multiply(const SignedLogReal<double> &a);
template void SignedLogReal<float>::Multiply(const SignedLogReal<float> &a);
template void SignedLogReal<double>::MultiplyReal(double f);
template void SignedLogReal<float>::MultiplyReal(float f);
template void SignedLogReal<double>::MultiplyLogReal(double f);
template void SignedLogReal<float>::MultiplyLogReal(float f);
template void SignedLogReal<double>::DivideBy(const SignedLogReal<double> &a);
template void SignedLogReal<float>::DivideBy(const SignedLogReal<float> &a);

template SignedLogReal<double> SignedLogReal<double>::operator+(
    const SignedLogReal<double> &a) const;
template SignedLogReal<double> SignedLogReal<double>::operator*(
    const SignedLogReal<double> &a) const ;
template SignedLogReal<double> SignedLogReal<double>::operator/(
    const SignedLogReal<double> &a) const;
template SignedLogReal<double> SignedLogReal<double>::operator-(
    const SignedLogReal<double> &a) const;
template SignedLogReal<double> operator-(const SignedLogReal<double> &a);

template SignedLogReal<double>::SignedLogReal(double f);
template SignedLogReal<double>::SignedLogReal(float f);
template SignedLogReal<float>::SignedLogReal(double f);
template SignedLogReal<float>::SignedLogReal(float f);

template SignedLogReal<double>::SignedLogReal(bool s, double);
template SignedLogReal<double>::SignedLogReal(bool s, float);
template SignedLogReal<float>::SignedLogReal(bool s, double);
template SignedLogReal<float>::SignedLogReal(bool s, float);

template SignedLogReal<double>::SignedLogReal(const SignedLogReal<double> &);
template SignedLogReal<double>::SignedLogReal(const SignedLogReal<float> &);
template SignedLogReal<float>::SignedLogReal(const SignedLogReal<double> &);
template SignedLogReal<float>::SignedLogReal(const SignedLogReal<float> &);

template class SignedLogReal<double>;
template class SignedLogReal<float>;

} // namespace kaldi
