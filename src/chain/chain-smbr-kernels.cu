// chain/chain-kernels.cu

// Copyright  2015  Johns Hopkins University (author: Daniel Povey)


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


#include <cfloat>
#include "chain/chain-kernels-ansi.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 200
#error - Kaldi no longer supports CC1.x devices. Please use a newer GPU or \
         configure with --use-cuda=no (this will disable the use of GPU).
#endif


#ifdef __CUDACC__
#if ( __CUDACC_VER_MAJOR__ >= 8 ) && ( !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600 )
// native implementation available
#else
#if __CUDA_ARCH__ >= 600
#error using CAS implementation of double atomicAdd
#endif
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif
#endif


template <typename Real>
__device__ inline void atomic_add(Real* address, Real value) {
  atomicAdd(address, value);
}

template <typename Real>
__device__ inline void atomic_add_thresholded(Real* address, Real value) {
  // This function uses a randomized algorithm to only do atomic adds for values
  // with absolute value >= a threshold, 
  // and if it's below the threshold, randomly add the
  // threshold itself with probability (value / threshold).  This preserves
  // expectations.  

  // kThresholdingPowerOfTwo is defined in chain-datastruct.h; it defines
  // the threshold for randomized posterior pruning.
  const Real threshold = 1.0 / (1 << kThresholdingPowerOfTwo);
  Real abs_value = abs(value);
  if (abs_value >= threshold) {
    atomic_add(address, value);
  } else {
    // The intention here is to do:
    // with probability(value / threshold), do:
    //   atomic_add(address, threshold);
    // We use the least significant bits of the value as a source of
    // randomness.  It would probably be more efficient to extract these
    // random bits directly from the float, but I don't want to have to
    // deal with endian-ness issues.
    //
    // below, x is a fixed-point representation of (value / threshold); it would
    // be 16777216 == 2^24 if value == threshold and 0 if value == 0.  We choose
    // the power 24 because that's the number of binary digits in the mantissa
    // in IEEE single precision floating point.
    // Note: we parenthesize the expression like this so that the
    // denominator can be precomputed as a constant expression.
    int32_cuda x = abs_value / (threshold / (1 << 24));
    // in the line below, the expression (x >> 12) is a representation of (value /
    // threshold) between 0 and 4096, with 4096 representing (value / threshold ==
    // 1), while (x & 4095) is treated as a pseudorandom number between 0 and 4095.
    if ((x >> 12) > (x & 4095)) {
      if (value >= 0) atomic_add(address, threshold);
      else atomic_add(address, -threshold);
    }
  }
}

// one iteration of the forward computation in the chain HMM with 
// SMBR objective.
// The grid y determines which HMM-state we handle.  [put this in the grid because
// HMM-states don't all take the same amount of time in the backwards direction, and it's
// better for scheduling to have them at the outer level.]
// The block x and grid x determine which sequence (0 ... num_sequences - 1) we handle;
// note that num_sequences == the number of elements in the minibatch, and we
// insist they all have the same number of time steps.
// note: 'probs' is indexed by sequence-index + (pdf-index * prob_stride).
// note: 'num_post' is indexed by sequence-index + (pdf-index * post_stride).
__global__
static void _cuda_chain_smbr_hmm_forward(
    const Int32Pair *backward_transitions,
    const DenominatorGraphTransition *transitions,
    int32_cuda num_sequences,
    int32_cuda num_hmm_states,
    const BaseFloat *probs, int32_cuda prob_stride, 
    const BaseFloat *num_post, int32_cuda post_stride,
    const BaseFloat *prev_alpha, const BaseFloat *prev_alpha_smbr,
    BaseFloat *this_alpha, BaseFloat *this_alpha_smbr) {
  // 'backward_transitions', indexed by hmm-state, consists of [start, end]
  // indexes into the 'transitions' array.  This gives us the info for
  // transitions *into* this state.  'probs' contains the exponentiated neural
  // net outputs; it has dimension num-output-indexes by num_sequences and its
  // stride is 'prob_stride'.  'prev_alpha' and 'this_alpha', which are
  // extracted from a larger matrix, both have dimension num-history-states by
  // num-sequences. 'prev_alpha_smbr' and 'this_alpha_smbr' are analogous 
  // for the partial SMBR values.

  // s is the index of the sequence within the minibatch,
  // from 0 .. num-egs-in-this-minibatch - 1.
  // h is the hmm-state index.
  int32_cuda s = threadIdx.x + blockIdx.x * blockDim.x,
      h  = blockIdx.y;
  if (s >= num_sequences)
    return;

  double this_tot_alpha = 0.0, this_tot_alpha_smbr = 0.0;
  const DenominatorGraphTransition
      *trans_iter = transitions + backward_transitions[h].first,
      *trans_end = transitions + backward_transitions[h].second;
  // Note: regarding this loop unrolling, I tried the automatic unrolling using
  // #pragma unroll 2 (after modifying the loop to have an integer index), but I
  // did not see any performance improvement, it was slightly slower.  So the
  // compiler must be doing something different than what I'm doing here.
  const int loop_unroll = 2;  // don't change this without changing the code
                              // below.
  for (; trans_iter + loop_unroll <= trans_end; trans_iter += loop_unroll) {
    BaseFloat transition_prob0 = trans_iter[0].transition_prob;
    int32_cuda pdf_id0 = trans_iter[0].pdf_id,
        prev_hmm_state0 = trans_iter[0].hmm_state;
    BaseFloat transition_prob1 = trans_iter[1].transition_prob;
    int32_cuda pdf_id1 = trans_iter[1].pdf_id,
        prev_hmm_state1 = trans_iter[1].hmm_state;
    BaseFloat pseudo_loglike0 = probs[pdf_id0 * prob_stride + s],
        num_post0 = num_post[pdf_id0 * post_stride + s],
        this_prev_alpha0 = prev_alpha[prev_hmm_state0 * num_sequences + s],
        this_prev_alpha_smbr0 = 
               prev_alpha_smbr[prev_hmm_state0 * num_sequences + s],
        pseudo_loglike1 = probs[pdf_id1 * prob_stride + s],
        num_post1 = num_post[pdf_id1 * post_stride + s],
        this_prev_alpha1 = prev_alpha[prev_hmm_state1 * num_sequences + s],
        this_prev_alpha_smbr1 =
          prev_alpha_smbr[prev_hmm_state1 * num_sequences + s];

    this_tot_alpha += this_prev_alpha0 * transition_prob0 * pseudo_loglike0 +
                       this_prev_alpha1 * transition_prob1 * pseudo_loglike1;
    this_tot_alpha_smbr += 
      (this_prev_alpha_smbr0 + num_post0) * this_prev_alpha0 
      * transition_prob0 * pseudo_loglike0
      + (this_prev_alpha_smbr1 + num_post1) * this_prev_alpha1 
      * transition_prob1 * pseudo_loglike1;
  }
  if (trans_iter != trans_end) {
    // mop up the odd transition.
    BaseFloat transition_prob0 = trans_iter[0].transition_prob;
    int32_cuda pdf_id0 = trans_iter[0].pdf_id,
       prev_hmm_state0 = trans_iter[0].hmm_state;
    BaseFloat pseudo_loglike0 = probs[pdf_id0 * prob_stride + s],
        num_post0 = num_post[pdf_id0 * post_stride + s],
        this_prev_alpha0 = prev_alpha[prev_hmm_state0 * num_sequences + s],
        this_prev_alpha_smbr0 = 
          prev_alpha_smbr[prev_hmm_state0 * num_sequences + s];
    this_tot_alpha += this_prev_alpha0 * transition_prob0 * pseudo_loglike0;
    this_tot_alpha_smbr += 
      (this_prev_alpha_smbr0 + num_post0) * this_prev_alpha0 
      * transition_prob0 * pseudo_loglike0;
  }

  // Let arbitrary_scale be the inverse of the sum of all alpha values on-- the
  // previous frame this sum of all the alpha values is stored in the place that
  // we'd store the previous alpha for state-index equal to num_hmm_states
  // (i.e. one past the end).  We multiply this into all the
  // transition-probabilities from the previous frame to this frame, in both the
  // forward and backward passes, in order to keep the alphas in a good numeric
  // range.  This won't affect the posteriors, as it's just a constant factor
  // for each frame, but when computing the total likelihood we'll need to
  // compensate for it later on.
  BaseFloat arbitrary_scale = 
      1.0 / prev_alpha[num_hmm_states * num_sequences + s];
  this_alpha[h * num_sequences + s] = this_tot_alpha * arbitrary_scale;
  if (this_tot_alpha > 0.0)
    this_alpha_smbr[h * num_sequences + s] = 
      this_tot_alpha_smbr / this_tot_alpha;
  else 
    this_alpha_smbr[h * num_sequences + s] = 0.0;
}


__global__
static void _cuda_chain_smbr_hmm_backward(
    const Int32Pair *forward_transitions,
    const DenominatorGraphTransition *transitions,
    int32_cuda num_sequences, int32_cuda num_hmm_states,
    const BaseFloat *probs, int32_cuda prob_stride, 
    const BaseFloat *num_post, int32_cuda post_stride,
    const BaseFloat *tot_smbr,
    const BaseFloat *this_alpha, const BaseFloat *this_alpha_smbr,
    const BaseFloat *next_beta, const BaseFloat *next_beta_smbr,
    BaseFloat *this_beta, BaseFloat *this_beta_smbr,
    BaseFloat *acc_deriv, int32_cuda acc_deriv_stride,
    BaseFloat *log_prob_deriv, int32_cuda log_prob_deriv_stride) {
  // 'forward_transitions', indexed by hmm-state, consists of [start, end]
  // indexes into the 'transition_info' array.  This is about the transitions
  // *out of* this state.  'probs' contains the exponentiated neural net
  // outputs; it has dimension num-output-indexes by num_sequences, and contains
  // just the observation probabilities for this time index.  Its stride is
  // prob_stride.
  // 'this_alpha', 'next_beta' and 'this_beta' all have dimension
  // num-history-states by num-sequences.
  // 'this_alpha_smbr', 'next_beta_smbr', and 'this_beta_smbr' are 
  // analogous quantities storing values for SMBR objective.
  // The beta probs are normalized in such a way (by multiplying by 1/(total-data-prob))
  // that to get occupation counts we don't need to multiply by 1/total-data-prob.
  // deriv_scale is a factor (e.g. -1.0 or -0.99) that we multiply these derivs by
  // while accumulating them.

  // s is the index of the sequence within the minibatch,
  // from 0 .. num-egs-in-this-minibatch - 1.
  // h is the hmm-state index.
  int32_cuda s = threadIdx.x + blockIdx.x * blockDim.x,
      h = blockIdx.y;
  if (s >= num_sequences)
    return;

  // See where arbitrary_scale is defined in the forward computation above, for
  // more explanation of inv_arbitrary_scale.
  BaseFloat this_alpha_prob = this_alpha[h * num_sequences + s],
      this_alpha_smbr_i = this_alpha_smbr[h * num_sequences + s],
      inv_arbitrary_scale =
      this_alpha[num_hmm_states * num_sequences + s];
  double tot_variable_factor = 0.0, tot_beta_smbr = 0.0;

  BaseFloat occupation_factor = this_alpha_prob / inv_arbitrary_scale;
  const DenominatorGraphTransition
      *trans_iter = transitions + forward_transitions[h].first,
      *trans_end = transitions + forward_transitions[h].second;
  const int loop_unroll = 2;  // don't change this without changing the code
                              // below.
  for (; trans_iter + loop_unroll <= trans_end; trans_iter += loop_unroll) {
    BaseFloat transition_prob0 = trans_iter[0].transition_prob;
    int32_cuda pdf_id0 = trans_iter[0].pdf_id,
        next_hmm_state0 = trans_iter[0].hmm_state;
    BaseFloat transition_prob1 = trans_iter[1].transition_prob;
    int32_cuda pdf_id1 = trans_iter[1].pdf_id,
        next_hmm_state1 = trans_iter[1].hmm_state;
    BaseFloat next_beta_j0 = next_beta[next_hmm_state0 * num_sequences + s],
        next_beta_smbr_j0 = next_beta_smbr[next_hmm_state0 * num_sequences + s],
        next_beta_j1 = next_beta[next_hmm_state1 * num_sequences + s],
        next_beta_smbr_j1 = next_beta_smbr[next_hmm_state1 * num_sequences + s],
        prob0 = probs[pdf_id0 * prob_stride + s],
        prob1 = probs[pdf_id1 * prob_stride + s],
        num_post0 = num_post[pdf_id0 * post_stride + s], 
        num_post1 = num_post[pdf_id1 * post_stride + s];

    BaseFloat variable_factor0 = transition_prob0 * next_beta_j0 * prob0,
        variable_factor1 = transition_prob1 * next_beta_j1 * prob1;
    tot_beta_smbr += (next_beta_smbr_j0 + num_post0) * variable_factor0
      + (next_beta_smbr_j1 + num_post1) * variable_factor1;
    tot_variable_factor += variable_factor0 + variable_factor1;
    BaseFloat occupation_prob0 = variable_factor0 * occupation_factor;
    BaseFloat this_acc_r0 = occupation_prob0 
      * (this_alpha_smbr_i + num_post0 + next_beta_smbr_j0);
    atomic_add(acc_deriv + (pdf_id0 * acc_deriv_stride + s),
               this_acc_r0);
    atomic_add(log_prob_deriv + (pdf_id0 * log_prob_deriv_stride + s),
               occupation_prob0);
    BaseFloat occupation_prob1 = variable_factor1 * occupation_factor;
    BaseFloat this_acc_r1 = occupation_prob1
      * (this_alpha_smbr_i + num_post1 + next_beta_smbr_j1);
    atomic_add(acc_deriv + (pdf_id1 * acc_deriv_stride + s),
               this_acc_r1);
    atomic_add(log_prob_deriv + (pdf_id1 * log_prob_deriv_stride + s),
               occupation_prob1);
  }
  if (trans_iter != trans_end) {
    // mop up the odd transition.
    BaseFloat transition_prob0 = trans_iter[0].transition_prob;
    int32_cuda pdf_id0 = trans_iter[0].pdf_id,
        next_hmm_state0 = trans_iter[0].hmm_state;
    BaseFloat next_beta_j0 = next_beta[next_hmm_state0 * num_sequences + s],
        next_beta_smbr_j0 = next_beta_smbr[next_hmm_state0 * num_sequences + s],
        prob0 = probs[pdf_id0 * prob_stride + s],
        num_post0 = num_post[pdf_id0 * post_stride + s];
    BaseFloat variable_factor0 = transition_prob0 * next_beta_j0 * prob0;
    tot_beta_smbr += (next_beta_smbr_j0 + num_post0) * variable_factor0;
    tot_variable_factor += variable_factor0;
    BaseFloat occupation_prob0 = variable_factor0 * occupation_factor;
    BaseFloat this_acc_r0 = occupation_prob0
      * (this_alpha_smbr_i + num_post0 + next_beta_smbr_j0);
    atomic_add(acc_deriv + (pdf_id0 * acc_deriv_stride + s),
               this_acc_r0);
    atomic_add(log_prob_deriv + (pdf_id0 * log_prob_deriv_stride + s),
               occupation_prob0);
  }
  BaseFloat beta = tot_variable_factor / inv_arbitrary_scale;
  this_beta[h * num_sequences + s] = beta;
  if (tot_variable_factor > 0.0)
    this_beta_smbr[h * num_sequences + s] = 
        tot_beta_smbr / tot_variable_factor;
  else
    this_beta_smbr[h * num_sequences + s] = 0.0;
}


// Chain forward with SMBR objective
void cuda_chain_smbr_hmm_forward(
    dim3 Gr, dim3 Bl,
    const Int32Pair *backward_transitions,
    const DenominatorGraphTransition *transitions,
    int32_cuda num_sequences,
    int32_cuda num_hmm_states,
    const BaseFloat *probs, int32_cuda prob_stride,
    const BaseFloat *num_post, int32_cuda post_stride,
    const BaseFloat *prev_alpha, const BaseFloat *prev_alpha_smbr,
    BaseFloat *this_alpha, BaseFloat *this_alpha_smbr) {
  _cuda_chain_smbr_hmm_forward<<<Gr,Bl>>>(
      backward_transitions, transitions,
      num_sequences, num_hmm_states,
      probs, prob_stride, num_post, post_stride,
      prev_alpha, prev_alpha_smbr, this_alpha, this_alpha_smbr);
}

void cuda_chain_smbr_hmm_backward(
    dim3 Gr, dim3 Bl,
    const Int32Pair *forward_transitions,
    const DenominatorGraphTransition *transitions,
    int32_cuda num_sequences,
    int32_cuda num_hmm_states,
    const BaseFloat *probs, int32_cuda prob_stride,
    const BaseFloat *num_post, int32_cuda post_stride,
    const BaseFloat *tot_smbr,
    const BaseFloat *this_alpha, const BaseFloat *this_alpha_smbr,
    const BaseFloat *next_beta, const BaseFloat *next_beta_smbr,
    BaseFloat *this_beta, BaseFloat *this_beta_smbr,
    BaseFloat *acc_deriv,
    int32_cuda acc_deriv_stride,
    BaseFloat *log_prob_deriv,
    int32_cuda log_prob_deriv_stride) {
  _cuda_chain_smbr_hmm_backward<<<Gr,Bl>>>(
      forward_transitions, transitions,
      num_sequences, num_hmm_states,
      probs, prob_stride, num_post, post_stride, tot_smbr,
      this_alpha, this_alpha_smbr, next_beta, next_beta_smbr,
      this_beta, this_beta_smbr, 
      acc_deriv, acc_deriv_stride, 
      log_prob_deriv, log_prob_deriv_stride);
}
