// base/kaldi-types-extra-test.cc

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

#include "base/kaldi-common.h"
#include "base/kaldi-types-extra.h"

namespace kaldi {
  template<typename Real>
  void UnitTestSignedLogRealConstructor() {
    Real f = RandGauss();
    SignedLogReal<Real> a(f);

    KALDI_ASSERT((f < 0.0 && a.Negative()) || (f >= 0.0 && a.Positive()));
    KALDI_ASSERT(kaldi::ApproxEqual(a.LogMagnitude(),
                 (f < 0.0 ? Log(-f) : Log(f))));
  }
  
  template<typename Real>
  void UnitTestSignedLogRealAdd() {
    Real f1 = RandGauss();
    Real f2 = RandGauss();

    SignedLogReal<Real> a1(f1);
    SignedLogReal<Real> a2(f2);
    
    SignedLogReal<Real> sum(f1 + f2);
 
    {
      SignedLogReal<Real> temp(a1);
      temp.Add(a2);

      KALDI_ASSERT(kaldi::ApproxEqual(temp.LogMagnitude(), sum.LogMagnitude()));
      KALDI_ASSERT(temp.ApproxEqual(sum));
    }

    {
      SignedLogReal<Real> temp(a1);
      temp.AddReal(f2);
      KALDI_ASSERT(kaldi::ApproxEqual(temp.LogMagnitude(), sum.LogMagnitude()));
      KALDI_ASSERT(temp.ApproxEqual(sum));
    }

    if (f2 > 0.0) {
      SignedLogReal<Real> temp(a1);
      temp.AddLogReal(Log(f2));

      KALDI_ASSERT(kaldi::ApproxEqual(temp.LogMagnitude(), sum.LogMagnitude()));
      KALDI_ASSERT(temp.ApproxEqual(sum));
    }

    {
      SignedLogReal<Real> temp = a1 + a2;
      KALDI_ASSERT(temp.ApproxEqual(sum));
    }
  }
  
  template<typename Real>
  void UnitTestSignedLogRealSub() {
    Real f1 = RandGauss();
    Real f2 = RandGauss();

    SignedLogReal<Real> a1(f1);
    SignedLogReal<Real> a2(f2);
    
    SignedLogReal<Real> diff(f1 - f2);
 
    {
      SignedLogReal<Real> temp(a1);
      temp.Sub(a2);

      KALDI_ASSERT(kaldi::ApproxEqual(temp.LogMagnitude(), diff.LogMagnitude()));
      KALDI_ASSERT(temp.ApproxEqual(diff));
    }

    {
      SignedLogReal<Real> temp(a1);
      temp.AddReal(-f2);
      KALDI_ASSERT(kaldi::ApproxEqual(temp.LogMagnitude(), diff.LogMagnitude()));
      KALDI_ASSERT(temp.ApproxEqual(diff));
    }

    if (f2 < 0.0) {
      SignedLogReal<Real> temp(a1);
      temp.AddLogReal(Log(-f2));

      KALDI_ASSERT(kaldi::ApproxEqual(temp.LogMagnitude(), diff.LogMagnitude()));
      KALDI_ASSERT(temp.ApproxEqual(diff));
    }
    
    {
      SignedLogReal<Real> temp = a1 - a2;
      KALDI_ASSERT(temp.ApproxEqual(diff));
    }
  }

  template<typename Real>
  void UnitTestSignedLogRealMultiply() {
    Real f1 = RandGauss();
    Real f2 = RandGauss();

    SignedLogReal<Real> a1(f1);
    SignedLogReal<Real> a2(f2);
    
    SignedLogReal<Real> product(f1 * f2);
  
    {
      SignedLogReal<Real> temp(a1);
      temp.Multiply(a2);

      KALDI_ASSERT(kaldi::ApproxEqual(temp.LogMagnitude(), product.LogMagnitude()));
      KALDI_ASSERT(temp.ApproxEqual(product));
    }

    {

      SignedLogReal<Real> temp(a1);
      temp.MultiplyReal(f2);
      KALDI_ASSERT(kaldi::ApproxEqual(temp.LogMagnitude(), product.LogMagnitude()));
      KALDI_ASSERT(temp.ApproxEqual(product));
    }

    if (f2 > 0.0) {
      SignedLogReal<Real> temp(a1);
      temp.MultiplyLogReal(Log(f2));

      KALDI_ASSERT(kaldi::ApproxEqual(temp.LogMagnitude(), product.LogMagnitude()));
      KALDI_ASSERT(temp.ApproxEqual(product));
    }
    
    {
      SignedLogReal<Real> temp = a1 * a2;
      KALDI_ASSERT(temp.ApproxEqual(product));
    }
  }

  template<typename Real>
  void UnitTestSignedLogRealEqual() {
    Real f = RandGauss();

    SignedLogReal<Real> a1(f);
    SignedLogReal<Real> a2(f);
    SignedLogReal<Real> a3(a2);
    SignedLogReal<Real> a4(f < 0.0, (f < 0.0) ? kaldi::Log(-f) : kaldi::Log(f));
    SignedLogReal<Real> a5;
    a5.Set(f);

    a1.Equal(a2);
    a1.Equal(a3);
    a1.Equal(a4);
    a1.Equal(a5);

    KALDI_ASSERT((f < 0.0 && a1.Negative()) || (f >= 0.0 && a1.Positive()));
    KALDI_ASSERT(kaldi::ApproxEqual(a1.LogMagnitude(),
                 (f < 0.0 ? Log(-f) : Log(f))));
  }

  template<typename Real>
  void UnitTestSignedLogRealApproxEqual() {
    Real f = RandGauss();
    Real f2 = f + 0.01 * f * RandGauss();
    SignedLogReal<Real> a1(f);
    SignedLogReal<Real> a2(f2);

    KALDI_ASSERT(kaldi::ApproxEqual(f, f2, 0.005) == a1.ApproxEqual(a2, 0.01));
  }

  template<typename Real>
  void UnitTestSignedLogRealIsOne() {
    Real f = 1.0 + 1.0e-20 * RandGauss();
    SignedLogReal<Real> a(f);
    KALDI_ASSERT(kaldi::ApproxEqual(f, 1.0, 1.0e-10) == a.IsOne(1e-10));
  }
  
  template<typename Real>
  void UnitTestSignedLogRealAddSubMultiplyScaled() {
    Real f1 = RandGauss();
    Real f2 = RandGauss();
    Real s = RandGauss();

    SignedLogReal<Real> a1(f1);
    SignedLogReal<Real> a2(f2);
    
    SignedLogReal<Real> sum(f1 + f2 * Exp(s));
    SignedLogReal<Real> diff(f1 - f2 * Exp(s));
  
    {
      SignedLogReal<Real> temp(a1);
      temp.AddMultiplyLogReal(a2, s);
      KALDI_ASSERT(temp.ApproxEqual(sum));
    }

    {
      SignedLogReal<Real> temp(a1);
      temp.SubMultiplyLogReal(a2, s);
      KALDI_ASSERT(temp.ApproxEqual(diff));
    }

    {
      SignedLogReal<Real> temp = a1 + a2 * SignedLogReal<Real>(false, s);
      KALDI_ASSERT(temp.ApproxEqual(sum));
    }
    
    {
      SignedLogReal<Real> temp = a1 + a2 * SignedLogReal<Real>(true, s);
      KALDI_ASSERT(temp.ApproxEqual(diff));
    }
    
    {
      SignedLogReal<Real> temp = a1 - a2 * SignedLogReal<Real>(false, s);
      KALDI_ASSERT(temp.ApproxEqual(diff));
    }
  }
}


int main() {
  using namespace kaldi;
  
  for (int i = 0; i < 100; i++) {
    UnitTestSignedLogRealConstructor<double>();
    UnitTestSignedLogRealAdd<double>();
    UnitTestSignedLogRealMultiply<double>();
    UnitTestSignedLogRealSub<double>();
    UnitTestSignedLogRealEqual<double>();
    UnitTestSignedLogRealAddSubMultiplyScaled<double>();
    UnitTestSignedLogRealApproxEqual<double>();
    UnitTestSignedLogRealIsOne<double>();
    UnitTestSignedLogRealConstructor<float>();
    UnitTestSignedLogRealAdd<float>();
    UnitTestSignedLogRealMultiply<float>();
    UnitTestSignedLogRealSub<float>();
    UnitTestSignedLogRealEqual<float>();
    UnitTestSignedLogRealAddSubMultiplyScaled<float>();
    UnitTestSignedLogRealApproxEqual<float>();
    UnitTestSignedLogRealIsOne<float>();
  }
}
