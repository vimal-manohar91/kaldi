// base/kaldi-types-extra.h

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

#ifndef KALDI_BASE_KALDI_TYPES_EXTRA_H_
#define KALDI_BASE_KALDI_TYPES_EXTRA_H_ 1

#include "base/kaldi-math.h"
#include "base/kaldi-types.h"
#include "base/kaldi-common.h"

namespace kaldi {

// This class is used to store a real number as pair of 
// sign and log-magnitude
template<typename Real>
class SignedLogReal {
  public:
    /// Returns the sign of the real number.
    /// true for negative, false for positive
    inline bool Sign() const { return sign_; }

    /// Returns true if the number is positive
    inline bool Positive() const { return (!sign_); }
    
    /// Returns true if the number is negative
    inline bool Negative() const { return sign_; }

    /// Returns the log magnitude of the real number
    inline Real LogMagnitude() const { return log_f_; }

    /// Returns the real number in double
    inline Real Value() const { return
      static_cast<Real>(Exp(static_cast<double>(log_f_)) * 
          (sign_ ? -1.0 : 1.0)); }

    /*    Basic setting-to-special values functions.    */

    /// Sets the value to zero
    void SetZero();

    /// Sets the number to particular value
    void Set(Real);

    /// Sets the number to one
    void SetOne();

    /// Sets the number to random value from normal distribution
    void SetRandn();

    /// Sets the number to uniformly distributed on (0,1)
    void SetRandUniform();

    /* Various special functions. */
    
    /// Negates the stored real number
    void Negate() { sign_ = !sign_; };

    /// Apply log to the stored real number, if positive;
    /// otherwise exit with error.
    void Log();
    
    /// Returns true if the number is zero upto an epsilon distance
    bool IsZero(Real cutoff = 1.0e-40) const;

    /// Returns true if the number is one upto an epsilon distance
    bool IsOne(Real cutoff = 1.0e-06) const;

    /// Returns true if this - other <= tol * this
    bool ApproxEqual(const SignedLogReal<Real> &other, float tol = 0.01) const;

    /// Tests for exact equality
    bool Equal(const SignedLogReal<Real> &other) const;

    /* Simple operations */

    /// Add another object of same type
    /// this->Value() += a.Value()
    template<typename OtherReal> void Add(const SignedLogReal<OtherReal> &a); 

    /// Add a real number
    /// this->Value() += f
    template<typename OtherReal> void AddReal(OtherReal f);

    /// Add a real number that is stored in log-domain
    /// this->Value() += Exp(log_f)
    template<typename OtherReal> void AddLogReal(OtherReal log_f);
    
    /// Add another SignedLogReal multiplied by a real number stored in
    /// log-domain
    /// this->Value() += a.Value() * Exp(log_f)
    template<typename OtherReal> void AddMultiplyLogReal(
        const SignedLogReal<OtherReal> &a, OtherReal log_f);
    
    /// Subtract another object of same type
    /// this->Value() -= a.Value()
    template<typename OtherReal> void Sub(const SignedLogReal<OtherReal> &a); 
    
    /// Subtract another SignedLogReal multiplied by a real number stored in
    /// log-domain
    /// this->Value() -= a.Value() * Exp(log_f)
    template<typename OtherReal> void SubMultiplyLogReal(
        const SignedLogReal<OtherReal> &a, OtherReal log_f);

    /// Multiply by another object of same type
    /// this->Value() *= a.Value() 
    template<typename OtherReal> void Multiply(const SignedLogReal<OtherReal> &a);

    /// Multiply by a real number
    /// this->Value() *= f
    template<typename OtherReal> void MultiplyReal(OtherReal f);
    
    /// Multiply by a real number stored in log-domain
    /// this->Value() *= Exp(f)
    template<typename OtherReal> void MultiplyLogReal(OtherReal log_f);
    
    /// DivideBy another object of same type
    /// this->Value() /= Exp(f)
    template<typename OtherReal> void DivideBy(const SignedLogReal<OtherReal> &a);

    /* Operators */
    /// These allow the objects of this class to work in the same
    /// way as the basic datatypes.
    /// These return a new object of the same class
   
    /// Returns an object whole Value() is this->Value() + a.Value()
    SignedLogReal<Real> operator+(const SignedLogReal<Real> &a) const;
    /// Returns an object whole Value() is this->Value() - a.Value()
    SignedLogReal<Real> operator-(const SignedLogReal<Real> &a) const;
    /// Returns an object whole Value() is this->Value() * a.Value()
    SignedLogReal<Real> operator*(const SignedLogReal<Real> &a) const;
    /// Returns an object whole Value() is this->Value() / a.Value()
    SignedLogReal<Real> operator/(const SignedLogReal<Real> &a) const;

    /* Initializers */

    /// Default initializer
    /// Initialize an object whose Value() is 0.0
    explicit SignedLogReal() :
      sign_(false), log_f_(kLogZeroDouble) { 
      KALDI_ASSERT_IS_FLOATING_TYPE(Real);
    }
  
    /// Initialize from a real number f
    /// Initializes an object whole Value() is f
    template<typename OtherReal>
    explicit SignedLogReal(OtherReal f) { 
      KALDI_ASSERT_IS_FLOATING_TYPE(Real);
      KALDI_ASSERT_IS_FLOATING_TYPE(OtherReal);
      if (f < 0.0) {
        sign_ = true;
        log_f_ = static_cast<Real>(kaldi::Log(static_cast<double>(-f)));
      } else {
        sign_ = false;
        log_f_ = static_cast<Real>(kaldi::Log(static_cast<double>(f)));
      }
    }

    /// Initialize from sign and log real number
    template<typename OtherReal>
    explicit SignedLogReal(bool sign, OtherReal log_f) :
      sign_(sign), log_f_(log_f) {
      KALDI_ASSERT_IS_FLOATING_TYPE(Real);
      KALDI_ASSERT_IS_FLOATING_TYPE(OtherReal);
    }

    /// Copy constructor
    template<typename OtherReal>
    explicit SignedLogReal(const SignedLogReal<OtherReal> &a) :
      sign_(a.Sign()), log_f_(a.LogMagnitude()) {
      KALDI_ASSERT_IS_FLOATING_TYPE(Real);
      KALDI_ASSERT_IS_FLOATING_TYPE(OtherReal);
    }

  private:
    bool sign_;   // true for negative numbers, false for positive
    Real log_f_;  // log-magnitude of stored number
};

/// ostream operator
/// Allows to write the object to output streams such as 
/// cout and KALDI_LOG
template<typename Real>
inline std::ostream & operator << (std::ostream & os, const SignedLogReal<Real> &a) {
  os << (a.Negative() ? "-" : "") << "1.0 * Exp(" << a.LogMagnitude() << ")";
  return os;
}

/// Unary operator '-'
/// Returns an object whole value is -a.Value()
template<typename Real>
SignedLogReal<Real> operator-(const SignedLogReal<Real> &a);

} // namespace kaldi

#endif  // KALDI_BASE_KALDI_TYPES_EXTRA_H_
