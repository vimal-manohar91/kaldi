// discriminative/discriminative-training.cc

// Copyright      2012-2015    Johns Hopkins University (author: Daniel Povey)
// Copyright      2014-2015    Vimal Manohar

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

#include "nnet3/discriminative-training.h"
#include "lat/lattice-functions.h"
#include "chain/chain-kernels-ansi.h"
#include "chain/chain-numerator.h"
#include "chain/chain-denominator.h"

namespace kaldi {
namespace chain {

void ComputeDiscriminativeObjfAndDeriv(const DiscriminativeTrainingOptions &opts,
                                       const TransitionModel &tmodel,
                                       const Vector<BaseFloat> &priors,
                                       const DiscriminativeSupervision &supervision,
                                       const CuMatrixBase<BaseFloat> &nnet_output,
                                       BaseFloat *tot_objf,
                                       BaseFloat *tot_weight,
                                       CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  DiscriminativeComputation computation(opts, tmodel, priors, supervision, nnet_output, tot_objf, tot_weight, nnet_output_dervi);
  computation.Compute();
}

class DiscriminativeComputation {
 public:
  DiscriminativeComputation(const DiscriminativeTrainingOptions &opts,
                            const TransitionModel &tmodel,
                            const Vector<BaseFloat> &priors,
                            const DiscriminativeSupervision &supervision,
                            const CuMatrixBase<BaseFloat> &nnet_output,
                            BaseFloat *tot_objf,
                            BaseFloat *tot_weight,
                            CuMatrixBase<BaseFloat> *nnet_output_deriv);

  void Compute();
 
 private:
  const DiscriminativeTrainingOptions &opts_;
  const TransitionModel &tmodel_;
  const Vector<BaseFloat> &priors_;
  const DiscriminativeSupervision_ &supervision_;
  const CuMatrixBase<BaseFloat> &nnet_output_;

  DiscriminiativeTrainingStats *stats_;
  CuMatrixBase<BaseFloat> *nnet_output_dervi_;

  Lattice num_lat_;
  bool num_lat_present;
  Lattice den_lat_;
};

DiscriminativeComputation::DiscriminativeComputation(
                            const DiscriminativeTrainingOptions &opts,
                            const TransitionModel &tmodel,
                            const Vector<BaseFloat> &priors,
                            const DiscriminativeSupervision &supervision,
                            const CuMatrixBase<BaseFloat> &nnet_output,
                            DiscriminativeTrainingStats *stats,
                            CuMatrixBase<BaseFloat> *nnet_output_deriv)
  : opts_(opts), tmodel_(tmodel), priors_(priors), 
  supervision_(supervision), nnet_output_(nnet_output),
  stats_(stats), nnet_output_deriv_(nnet_output_deriv), 
  num_lat_present_(supervision.num_lat_present) {
  if (num_lat_present_) {
    num_lat_ = supervision.num_lat;
    TopSort(&num_lat_);
  }
  
  den_lat_ = supervision.den_lat;
  TopSort(&den_lat_);
  
  if (!SplitStringToIntegers(opts_.silence_phones_str, ":", false,
                             &silence_phones_)) {
    KALDI_ERR << "Bad value for --silence-phones option: "
              << opts_.silence_phones_str;
  }
}

void Compute() {
  if (opts_.criterion == "mmi" && opts_.boost != 0.0) {
    BaseFloat max_silence_error = 0.0;
    LatticeBoost(tmodel_, supervision_.num_ali, silence_phones_,
                 opts_.boost, max_silence_error, &den_lat_);
  }

  int32 num_frames = supervision_.frames_per_sequence * supervision_.num_frames;
  
  if (stats_) {
    stats_->tot_t += num_frames;
    stats_->tot_t_weighted += num_frames * supervision_.weight;
  }

  int32 num_pdfs = nnet_output_.NumCols();
  KALDI_ASSERT(num_pdfs == priors_.Dim());
  
  // We need to look up the posteriors of some pdf-ids in the matrix
  // "posteriors".  Rather than looking them all up using operator (), which is
  // very slow because each lookup involves a separate CUDA call with
  // communication over PciExpress, we look them up all at once using
  // CuMatrix::Lookup().
  
  std::vector<Int32Pair> requested_indexes;
  BaseFloat wiggle_room = 1.3; // value not critical.. it's just 'reserve'
  
  int32 num_reserve = wiggle_room * den_lat_.NumStates();
  
  if (opts_.criterion == "mmi") {
    // For looking up the posteriors corresponding to the pdfs in the alignment
    num_reserve += num_frames;
  } else if (opts_.criterion == "nce" || opts_.criterion == "empfe" || 
             opts_.criterion == "esmbr") {
    // For looking up the posteriors corresponding to the pdfs in the 
    // numerator lattice or the numerator posteriors
    if (supervision_.num_lat_present) num_reserve *= 2;
  }

  requested_indexes.reserve(num_reserve);
  
  // Denominator probabilities to look up from denominator lattice
  std::vector<int32> state_times;
  int32 T = LatticeStateTimes(den_lat_, &state_times);
  KALDI_ASSERT(T == num_frames);
  
  StateId num_states = den_lat_.NumStates();
  for (StateId s = 0; s < num_states; s++) {
    StateId t = state_times[s];
    for (fst::ArcIterator<Lattice> aiter(den_lat_, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) { // input-side has transition-ids, output-side empty
        int32 tid = arc.ilabel, pdf_id = tmodel_.TransitionIdToPdf(tid);
        requested_indexes.push_back(MakePair(t, pdf_id));
      }
    }
  }

  if (opts_.criterion == "mmi") {
    // Numerator probabilities to look up from alignment
    for (int32 t = 0; t < num_frames; t++) {
      int32 tid = supervision_.num_ali[t], pdf_id = tmodel_.TransitionIdToPdf(tid);
      KALDI_ASSERT(pdf_id >= 0 && pdf_id < num_pdfs);
      requested_indexes.push_back(MakePair(t, pdf_id));
    }
  } else if (opts_.criterion == "nce" || opts_.criterion == "empfe" || opts_.criterion == "esmbr") {
    if (supervision_.num_lat_present) {
      // Numerator probabilities to look up from numerator lattice
      std::vector<int32> state_times;
      int32 T = LatticeStateTimes(num_lat_, &state_times);
      KALDI_ASSERT(T == num_frames);

      StateId num_states = num_lat_.NumStates();
      for (StateId s = 0; s < num_states; s++) {
        StateId t = state_times[s];
        for (fst::ArcIterator<Lattice> aiter(num_lat_, s); !aiter.Done(); aiter.Next()) {
          const Arc &arc = aiter.Value();
          if (arc.ilabel != 0) { // input-side has transition-ids, output-side empty
            int32 tid = arc.ilabel, pdf_id = tmodel_.TransitionIdToPdf(tid);
            requested_indexes.push_back(MakePair(t, pdf_id));
          }
        }
      }
    }
  }
  
  std::vector<BaseFloat> answers;
  CuArray<Int32Pair> cu_requested_indexes(requested_indexes);
  answers.resize(requested_indexes.size());
  posteriors.Lookup(cu_requested_indexes, &(answers[0]));
  // requested_indexes now contain (t, j) pair and answers contains the 
  // corresponding p(j|x(t)) as given by the neural network
  
  int32 num_floored = 0;

  BaseFloat floor_val = 1.0e-20; // floor for posteriors.
  size_t index;
  
  // Replace "answers" with the vector of scaled log-probs.  If this step takes
  // too much time, we can look at other ways to do it, using the CUDA card.
  for (index = 0; index < answers.size(); index++) {
    BaseFloat post = answers[index];
    if (post < floor_val) {
      post = floor_val;
      num_floored++;
    }
    int32 pdf_id = requested_indexes[index].second;
    BaseFloat pseudo_loglike = Log(post / priors(pdf_id)) * opts_.acoustic_scale;
    KALDI_ASSERT(!KALDI_ISINF(pseudo_loglike) && !KALDI_ISNAN(pseudo_loglike));
    answers[index] = pseudo_loglike;
  }


  BaseFloat num_logprob_weighted;
  if (nnet_output_deriv)
    nnet_output_deriv->SetZero();
  {
    NumeratorComputation numerator(supervision, nnet_output);
    // note: supervision.weight is included as a factor in the derivative from
    // the numerator object, and the logprob too.
    num_logprob_weighted = numerator.Forward();
    if (nnet_output_deriv)
      numerator.Backward(nnet_output_deriv);
  }
  DenominatorComputation denominator(opts, den_graph,
                                     supervision.num_sequences,
                                     nnet_output);

  BaseFloat den_logprob = denominator.Forward();
  if (nnet_output_deriv)
    denominator.Backward(-supervision.weight,
                         nnet_output_deriv);

  *tot_objf = num_logprob_weighted - supervision.weight * den_logprob;
  *tot_weight = supervision.weight * supervision.num_sequences *
      supervision.frames_per_sequence;
  if (!(*tot_objf  == *tot_objf)) {
    // inf or NaN detected
    if (nnet_output_deriv)
      nnet_output_deriv->SetZero();
    BaseFloat default_objf = -10;
    KALDI_WARN << "Objective function is " << (*tot_objf)
               << ", setting to " << default_objf << " per frame.";
    *tot_objf  = default_objf * *tot_weight;
  }

  // This code helps us see how big the derivatives are, on average,
  // for different frames of the sequences.  As expected, they are
  // smaller towards the edges of the sequences (due to the penalization
  // of 'incorrect' pdf-ids.
  if (GetVerboseLevel() >= 1) {
    int32 tot_frames = nnet_output_deriv->NumRows(),
 frames_per_sequence = supervision.frames_per_sequence,
       num_sequences = supervision.num_sequences;
    CuVector<BaseFloat> row_products(tot_frames);
    row_products.AddDiagMat2(1.0, *nnet_output_deriv, kNoTrans, 0.0);
    Vector<BaseFloat> row_products_cpu(row_products);
    Vector<BaseFloat> row_products_per_frame(frames_per_sequence);
    for (int32 i = 0; i < tot_frames; i++)
      row_products_per_frame(i / num_sequences) += row_products_cpu(i);
    KALDI_LOG << "Derivs per frame are " << row_products_per_frame;
  }
}


}  // namespace chain
}  // namespace kaldi

