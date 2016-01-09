// nnet3/discriminative-training.cc

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
#include "cudamatrix/cu-matrix.h"

namespace kaldi {
namespace discriminative {

class DiscriminativeComputation {
  typedef Lattice::Arc Arc;
  typedef Arc::StateId StateId;
 
 public:
  DiscriminativeComputation(const DiscriminativeTrainingOptions &opts,
                            const TransitionModel &tmodel,
                            const CuVectorBase<BaseFloat> &log_priors,
                            const DiscriminativeSupervision &supervision,
                            const CuMatrixBase<BaseFloat> &nnet_output,
                            DiscriminativeTrainingStats *stats,
                            CuMatrixBase<BaseFloat> *nnet_output_deriv);

  void Compute();
 
 private:
  const DiscriminativeTrainingOptions &opts_;
  const TransitionModel &tmodel_;
  const CuVectorBase<BaseFloat> &log_priors_;
  const DiscriminativeSupervision &supervision_;
  const CuMatrixBase<BaseFloat> &nnet_output_;

  DiscriminativeTrainingStats *stats_;
  CuMatrixBase<BaseFloat> *nnet_output_deriv_;

  Lattice num_lat_;
  bool num_lat_present_;
  Lattice den_lat_;

  std::vector<int32> silence_phones_;

  double ComputeObjfAndDeriv(Posterior *post);
  static inline Int32Pair MakePair(int32 first, int32 second) {
    Int32Pair ans;
    ans.first = first;
    ans.second = second;
    return ans;
  }
};

DiscriminativeComputation::DiscriminativeComputation(
                            const DiscriminativeTrainingOptions &opts,
                            const TransitionModel &tmodel,
                            const CuVectorBase<BaseFloat> &log_priors,
                            const DiscriminativeSupervision &supervision,
                            const CuMatrixBase<BaseFloat> &nnet_output,
                            DiscriminativeTrainingStats *stats,
                            CuMatrixBase<BaseFloat> *nnet_output_deriv)
  : opts_(opts), tmodel_(tmodel), log_priors_(log_priors), 
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

void DiscriminativeComputation::Compute() {
  if (opts_.criterion == "mmi" && opts_.boost != 0.0) {
    BaseFloat max_silence_error = 0.0;
    LatticeBoost(tmodel_, supervision_.num_ali, silence_phones_,
                 opts_.boost, max_silence_error, &den_lat_);
  }

  int32 num_frames = supervision_.frames_per_sequence * supervision_.num_sequences;
  
  int32 num_pdfs = nnet_output_.NumCols();
  KALDI_ASSERT(num_pdfs == log_priors_.Dim());
  
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
  nnet_output_.Lookup(cu_requested_indexes, &(answers[0]));
  // requested_indexes now contain (t, j) pair and answers contains the 
  // corresponding log p(j|x(t)) as given by the neural network
  
  int32 num_floored = 0;

  BaseFloat floor_val = -20 * kaldi::Log(10.0); // floor for posteriors.
  size_t index;
  
  // Replace "answers" with the vector of scaled log-probs.  If this step takes
  // too much time, we can look at other ways to do it, using the CUDA card.
  for (index = 0; index < answers.size(); index++) {
    BaseFloat log_post = answers[index];
    if (log_post < floor_val) {
      log_post = floor_val;
      num_floored++;
    }
    int32 pdf_id = requested_indexes[index].second;
    KALDI_ASSERT(log_post < 0 && log_priors_(pdf_id) < 0);
    BaseFloat pseudo_loglike = (log_post - log_priors_(pdf_id)) * opts_.acoustic_scale;
    KALDI_ASSERT(!KALDI_ISINF(pseudo_loglike) && !KALDI_ISNAN(pseudo_loglike));
    answers[index] = pseudo_loglike;
  }
  
  if (num_floored > 0) {
    KALDI_WARN << "Floored " << num_floored << " probabilities from nnet.";
  }

  index = 0;
  
  // Now put the negative (scaled) acoustic log-likelihoods in the lattice.
  for (StateId s = 0; s < num_states; s++) {
    for (fst::MutableArcIterator<Lattice> aiter(&den_lat_, s);
         !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel != 0) { // input-side has transition-ids, output-side empty
        arc.weight.SetValue2(-answers[index]);
        index++;
        aiter.SetValue(arc);
      }
    }
    LatticeWeight final = den_lat_.Final(s);
    if (final != LatticeWeight::Zero()) {
      final.SetValue2(0.0); // make sure no acoustic term in final-prob.
      den_lat_.SetFinal(s, final);
    }
  }
  
  DiscriminativeTrainingStats this_stats;
  if (stats_) 
    this_stats.SetConfig(stats_->config);
  
  // Look up numerator probabilities corresponding to alignment
  if (opts_.criterion == "mmi") {
    double tot_num_like = 0.0;
    KALDI_ASSERT(index + supervision_.num_ali.size() == answers.size());
    for (size_t this_index = 0; this_index < supervision_.num_ali.size(); this_index++)
      tot_num_like += answers[index + this_index];
    //KALDI_ASSERT(tot_num_like > 0); // In general, this must be positive because log_post is larger than log_prior for the correct labels
    this_stats.tot_num_objf += supervision_.weight * tot_num_like;
    index += supervision_.num_ali.size();
  } else if (opts_.criterion == "nce" || opts_.criterion == "empfe" || opts_.criterion == "esmbr") {
    if (supervision_.num_lat_present) {
      // Now put the negative (scaled) acoustic log-likelihoods in the 
      // numerator lattice.
      for (StateId s = 0; s < num_lat_.NumStates(); s++) {
        for (fst::MutableArcIterator<Lattice> aiter(&num_lat_, s);
            !aiter.Done(); aiter.Next()) {
          Arc arc = aiter.Value();
          if (arc.ilabel != 0) { // input-side has transition-ids, output-side empty
            arc.weight.SetValue2(-answers[index]);
            index++;
            aiter.SetValue(arc);
          }
        }
        LatticeWeight final = num_lat_.Final(s);
        if (final != LatticeWeight::Zero()) {
          final.SetValue2(0.0); // make sure no acoustic term in final-prob.
          num_lat_.SetFinal(s, final);
        }
      }
    } 
  }

  KALDI_ASSERT(index == answers.size());
  
  if (nnet_output_deriv_) {
    nnet_output_deriv_->SetZero();
    KALDI_ASSERT(nnet_output_deriv_->NumRows() == nnet_output_.NumRows() &&
        nnet_output_deriv_->NumCols() == nnet_output_.NumCols());
  }

  Posterior post;
  double objf = ComputeObjfAndDeriv(&post);
  
  //if (opts_.criterion == "mmi") {
  //  //KALDI_ASSERT(objf > 0); // Does not have to be necessarily true, but usually it is because on an average the log_post of states in the lattice is larger than log_prior
  //}
  this_stats.tot_objf += supervision_.weight * objf;
  
  KALDI_ASSERT(nnet_output_.NumRows() == post.size());
  
  if (nnet_output_deriv_) {
    SparseMatrix<BaseFloat> sp_output_deriv(nnet_output_.NumCols(), post);
    GeneralMatrix gen_output_deriv;
    gen_output_deriv.SwapSparseMatrix(&sp_output_deriv);
    gen_output_deriv.CopyToMat(nnet_output_deriv_, kNoTrans);
    if (supervision_.weight != 1.0)
      nnet_output_deriv_->Scale(supervision_.weight);
  }

  double tot_num_post = 0.0, tot_post = 0.0, tot_den_post = 0.0;

  if (nnet_output_deriv_) {
    if (opts_.criterion != "nce") {
      CuMatrix<BaseFloat> cu_post(*nnet_output_deriv_);
      cu_post.ApplyFloor(0.0);
      tot_num_post = cu_post.Sum();
      cu_post.CopyFromMat(*nnet_output_deriv_);
      cu_post.ApplyCeiling(0.0);
      tot_den_post = -cu_post.Sum();
    } else {
      CuMatrix<BaseFloat> cu_post(*nnet_output_deriv_);
      cu_post.ApplySignum();
      tot_post = cu_post.Sum();
    }

  //for (int32 t = 0; t < post.size(); t++) {
  //  for (int32 i = 0; i < post[t].size(); i++) {
  //    int32 pdf_id = post[t][i].first;
  //    // TODO: Check if the gradients are wrt to output correctly
  //    if (this_stats.AccumulateCounts())
  //      this_stats.indication_counts(pdf_id) += 1.0;
  //    BaseFloat weight = post[t][i].second;
  //    if (nnet_output_deriv_)
  //      (*nnet_output_deriv_)(t,pdf_id) = weight;
  //    if (opts_.criterion != "nce") {
  //      if (weight > 0.0) { tot_num_post += weight; }
  //      else { tot_den_post -= weight; }
  //    } else {
  //      tot_post += (weight > 0.0 ? weight: - weight);
  //    }
  //  }
  //}

    this_stats.tot_gradients += tot_post;
    this_stats.tot_den_count += tot_den_post;
    this_stats.tot_num_count += tot_num_post;
    
    if (this_stats.AccumulateGradients()) 
      (this_stats.gradients).AddRowSumMat(1.0, CuMatrix<double>(*nnet_output_deriv_));
    if (this_stats.AccumulateOutput()) {
      CuMatrix<double> temp(nnet_output_);
      temp.ApplyExp();
      (this_stats.output).AddRowSumMat(1.0, temp);
    }
  }
  
  this_stats.tot_t = T;
  this_stats.tot_t_weighted = T * supervision_.weight;
  
  if (!(this_stats.TotalObjf(opts_.criterion) == this_stats.TotalObjf(opts_.criterion))) {
    // inf or NaN detected
    if (nnet_output_deriv_)
      nnet_output_deriv_->SetZero();
    BaseFloat default_objf = -10;
    KALDI_WARN << "Objective function is " << this_stats.TotalObjf(opts_.criterion)
               << ", setting to " << default_objf << " per frame.";
    this_stats.tot_objf = default_objf * this_stats.tot_t_weighted;
  }
  
  if (GetVerboseLevel() >= 2) {
    if (GetVerboseLevel() >= 3) {
      this_stats.Print(opts_.criterion, true, true, true);
    } else 
      this_stats.Print(opts_.criterion);

  }

  if (stats_)
    stats_->Add(this_stats);

  // This code helps us see how big the derivatives are, on average,
  // for different frames of the sequences.  As expected, they are
  // smaller towards the edges of the sequences (due to the penalization
  // of 'incorrect' pdf-ids.
  if (nnet_output_deriv_ && GetVerboseLevel() >= 1) {
    int32 tot_frames = nnet_output_deriv_->NumRows(),
 frames_per_sequence = supervision_.frames_per_sequence,
       num_sequences = supervision_.num_sequences;
    CuVector<BaseFloat> row_products(tot_frames);
    row_products.AddDiagMat2(1.0, *nnet_output_deriv_, kNoTrans, 0.0);
    Vector<BaseFloat> row_products_cpu(row_products);
    Vector<BaseFloat> row_products_per_frame(frames_per_sequence);
    for (int32 i = 0; i < tot_frames; i++)
      row_products_per_frame(i / num_sequences) += row_products_cpu(i);
    KALDI_LOG << "Derivs per frame are " << row_products_per_frame;
  }
}

double DiscriminativeComputation::ComputeObjfAndDeriv(Posterior *post) {
  if (opts_.criterion == "mpfe" || opts_.criterion == "smbr") {
    Posterior tid_post;
    double ans = LatticeForwardBackwardMpeVariants(tmodel_, silence_phones_, den_lat_,
        supervision_.num_ali, opts_.criterion,
        opts_.one_silence_class,
        &tid_post);
    ConvertPosteriorToPdfs(tmodel_, tid_post, post);
    return ans;
  } else if (opts_.criterion == "mmi") {
    bool convert_to_pdfs = true, cancel = true;
    // we'll return the denominator-lattice forward backward likelihood,
    // which is one term in the objective function.
    return (LatticeForwardBackwardMmi(tmodel_, den_lat_, supervision_.num_ali,
                                      opts_.drop_frames, convert_to_pdfs,
                                      cancel, post));
  } else if (opts_.criterion == "nce") {
    Posterior tid_post;
    SignedLogDouble obj_func;

    if (supervision_.weights.size() > 0)
      obj_func = LatticeForwardBackwardNce(tmodel_, den_lat_, &tid_post, &supervision_.weights, opts_.weight_threshold);
    else
      obj_func = LatticeForwardBackwardNce(tmodel_, den_lat_, &tid_post);

    ConvertPosteriorToPdfs(tmodel_, tid_post, post);
    return obj_func.Value(); // returns the objective function.
  } else if (opts_.criterion == "empfe" || opts_.criterion == "esmbr") {
    double obj_func;
    Posterior tid_post;
    
    for (int32 debug_run = 0; debug_run <= 1; debug_run++) {
      if (opts_.debug_training) {
        KALDI_ASSERT(supervision_.oracle_ali.size() > 0);
        Posterior oracle_post;
        std::vector<std::string> debug_criteria;

        if (opts_.debug_training_advanced) {
          debug_criteria.resize(2);
          debug_criteria[0] = "smbr";
          debug_criteria[1] = "mpfe";
          debug_criteria.push_back("empfe");
          debug_criteria.push_back("esmbr");
        }

        double pdf_accuracy = 0.0, weighted_pdf_accuracy = 0.0;
        double phone_accuracy = 0.0, weighted_phone_accuracy = 0.0;

        for (size_t i = 0; i < supervision_.NumFrames(); i++) {
          phone_accuracy += ( tmodel_.TransitionIdToPhone(supervision_.num_ali[i]) == tmodel_.TransitionIdToPhone(supervision_.oracle_ali[i]) );
          pdf_accuracy += ( tmodel_.TransitionIdToPdf(supervision_.num_ali[i]) == tmodel_.TransitionIdToPdf(supervision_.oracle_ali[i]) );

          weighted_phone_accuracy += supervision_.weights[i] * ( tmodel_.TransitionIdToPhone(supervision_.num_ali[i]) == tmodel_.TransitionIdToPhone(supervision_.oracle_ali[i]) ) + (1 - supervision_.weights[i]) * ( tmodel_.TransitionIdToPhone(supervision_.num_ali[i]) != tmodel_.TransitionIdToPhone(supervision_.oracle_ali[i]) );
          weighted_pdf_accuracy += supervision_.weights[i] * ( tmodel_.TransitionIdToPdf(supervision_.num_ali[i]) == tmodel_.TransitionIdToPdf(supervision_.oracle_ali[i]) ) + (1 - supervision_.weights[i]) * ( tmodel_.TransitionIdToPdf(supervision_.num_ali[i]) != tmodel_.TransitionIdToPdf(supervision_.oracle_ali[i]) );
        }

        double expected_pdf_accuracy = 
          LatticeForwardBackwardEmpeVariants(tmodel_, 
              silence_phones_, num_lat_, supervision_.oracle_ali, NULL, NULL,
              "smbr", false, 0, 
              &oracle_post, 0.0);
        double expected_phone_accuracy = 
          LatticeForwardBackwardEmpeVariants(tmodel_, 
              silence_phones_, num_lat_, supervision_.oracle_ali, NULL, NULL,
              "mpfe", false, 0, 
              &oracle_post, 0.0);

        for (std::vector<std::string>::const_iterator it = debug_criteria.begin();
            it != debug_criteria.end(); ++it) {
          double obj_func_best_path = 
            LatticeForwardBackwardEmpeVariants(tmodel_, 
                silence_phones_, den_lat_, supervision_.num_ali, NULL, &num_lat_,
                *it, false, 0, 
                &oracle_post, 0.0);
          double obj_func_best_path_weighted = 
            LatticeForwardBackwardEmpeVariants(tmodel_, 
                silence_phones_, den_lat_, supervision_.num_ali, NULL, &num_lat_,
                *it, false, 0, 
                &oracle_post, 0.0, &supervision_.weights);
          double obj_func_oracle = 
            LatticeForwardBackwardEmpeVariants(tmodel_, 
                silence_phones_, den_lat_, supervision_.oracle_ali, NULL, &num_lat_,
                *it, false, 0, 
                &oracle_post, 0.0);

          KALDI_LOG << "self-training " << *it 
            << ": " << obj_func_best_path / supervision_.NumFrames()
            << "; weighted self-training " << *it 
            << ": " << obj_func_best_path_weighted / supervision_.NumFrames();
          if (*it == "smbr" || *it == "mpfe")
            KALDI_LOG << "oracle " << *it 
              << ": " << obj_func_oracle / supervision_.NumFrames();
        }
        KALDI_LOG << "pdf accuracy : " << pdf_accuracy / supervision_.NumFrames()
          << "; phone accuracy : " << phone_accuracy / supervision_.NumFrames();
        KALDI_LOG << "weighted pdf accuracy : " << weighted_pdf_accuracy / supervision_.NumFrames()
          << "; weighted phone accuracy : " << weighted_phone_accuracy / supervision_.NumFrames();
        KALDI_LOG << "expected pdf accuracy : " << expected_pdf_accuracy / supervision_.NumFrames()
          << "; expected phone accuracy : " << expected_phone_accuracy / supervision_.NumFrames();
      }

      if (debug_run == 0) {
        /*if (!supervision_.num_lat_present && supervision_.num_post.size() > 0) {
          // Using numerator posteriors
          obj_func = LatticeForwardBackwardEmpeVariants(tmodel_, 
                silence_phones_, den_lat_, supervision_.num_ali, &supervision_.num_post, NULL,
                opts_.criterion, opts_.one_silence_class, opts_.deletion_penalty, 
                &tid_post, opts_.weight_threshold);
        } else */ if (supervision_.num_lat_present) {
          // Using numerator lattice
          obj_func = LatticeForwardBackwardEmpeVariants(tmodel_, 
                silence_phones_, den_lat_, supervision_.num_ali, NULL, &num_lat_, 
                opts_.criterion, opts_.one_silence_class, opts_.deletion_penalty, 
                &tid_post, opts_.weight_threshold);
        } else {
          // Using denominator lattice
          obj_func = LatticeForwardBackwardEmpeVariants(tmodel_, 
                silence_phones_, den_lat_, supervision_.num_ali, NULL, NULL,
                opts_.criterion, opts_.one_silence_class, opts_.deletion_penalty, 
                &tid_post, opts_.weight_threshold);
        }
        ConvertPosteriorToPdfs(tmodel_, tid_post, post);
        KALDI_ASSERT(post->size() == supervision_.NumFrames());
        if (opts_.debug_training)
          KALDI_LOG << opts_.criterion << ": " << obj_func / supervision_.NumFrames();
      }
    }

    return obj_func;

  } else {
    KALDI_ERR << "Unknown criterion " << opts_.criterion;
  }

  return 0;
}


void ComputeDiscriminativeObjfAndDeriv(const DiscriminativeTrainingOptions &opts,
                                       const TransitionModel &tmodel,
                                       const CuVectorBase<BaseFloat> &log_priors,
                                       const DiscriminativeSupervision &supervision,
                                       const CuMatrixBase<BaseFloat> &nnet_output,
                                       DiscriminativeTrainingStats *stats,
                                       CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  DiscriminativeComputation computation(opts, tmodel, log_priors, supervision, nnet_output, stats, nnet_output_deriv);
  computation.Compute();
}

void DiscriminativeTrainingStats::Add(const DiscriminativeTrainingStats &other) {
  tot_t += other.tot_t;
  tot_t_weighted += other.tot_t_weighted;
  tot_objf += other.tot_objf;             // Actually tot_den_objf for mmi
  tot_gradients += other.tot_gradients;   // Only for nce 
  tot_num_count += other.tot_num_count;   // Not for nce
  tot_den_count += other.tot_den_count;   // Not for nce
  tot_num_objf += other.tot_num_objf;     // Only for mmi
  
  if (AccumulateGradients()) {
    gradients.AddVec(1.0, other.gradients);
  } 
  if (AccumulateOutput()) {
    output.AddVec(1.0, other.output);
  }
  if (AccumulateCounts()) {
    indication_counts.AddVec(1.0, other.indication_counts);
  }
}

void DiscriminativeTrainingStats::Print(const std::string &criterion, 
                                    bool print_avg_gradients, 
                                    bool print_avg_output,
                                    bool print_avg_counts) const {
  if (criterion == "mmi") {
    double num_objf = tot_num_objf / tot_t_weighted,
           den_objf = tot_objf / tot_t_weighted;
    double objf = num_objf - den_objf;

    double avg_post_per_frame = tot_num_count / tot_t_weighted;

    KALDI_LOG << "Number of frames is " << tot_t
              << " (weighted: " << tot_t_weighted
              << "), average (num or den) posterior per frame is "
              << avg_post_per_frame;
    KALDI_LOG << "MMI objective function is " << num_objf << " - "
              << den_objf << " = " << objf << " per frame, over "
              << tot_t_weighted << " frames.";
  } else if (criterion == "mpfe") {
    double avg_gradients = (tot_num_count + tot_den_count) / tot_t_weighted;
    double objf = tot_objf / tot_t_weighted;
    KALDI_LOG << "Average modulus of MPFE gradients is " << avg_gradients 
              << " per frame, over "
              << tot_t_weighted << " frames";
    KALDI_LOG << "MPFE objective function is " << objf
              << " per frame, over " << tot_t_weighted << " frames.";
  } else if (criterion == "smbr") {
    double avg_gradients = (tot_num_count + tot_den_count) / tot_t_weighted;
    double objf = tot_objf / tot_t_weighted;
    KALDI_LOG << "Average modulus of SMBR gradients is " << avg_gradients 
              << " per frame, over "
              << tot_t_weighted << " frames";
    KALDI_LOG << "SMBR objective function is " << objf
              << " per frame, over " << tot_t_weighted << " frames.";
  } else if (criterion == "nce") {
    double avg_gradients = (tot_gradients) / tot_t_weighted;
    double objf = tot_objf / tot_t_weighted;
    KALDI_LOG << "Average modulus of NCE gradients is " << avg_gradients 
              << " per frame, over "
              << tot_t_weighted << " frames";
    KALDI_LOG << "NCE objective function is " << objf << " per frame, over "
              << tot_t_weighted << " frames";
  } else if (criterion == "esmbr") {
    double avg_gradients = (tot_num_count + tot_den_count) / tot_t_weighted;
    double objf = tot_objf / tot_t_weighted;
    KALDI_LOG << "Average modulus of ESMBR gradients is " << avg_gradients 
              << " per frame, over "
              << tot_t_weighted << " frames";
    KALDI_LOG << "ESMBR objective function is " << objf << " per frame, over "
              << tot_t_weighted << " frames";
  } else if (criterion == "empfe") {
    double avg_gradients = (tot_num_count + tot_den_count) / tot_t_weighted;
    double objf = tot_objf / tot_t_weighted;
    KALDI_LOG << "Average modulus of EMPFE gradients is " << avg_gradients 
              << " per frame, over "
              << tot_t_weighted << " frames";
    KALDI_LOG << "EMPFE objective function is " << objf << " per frame, over "
              << tot_t_weighted << " frames";
  }
  
  if (AccumulateGradients()) {
    {
      Vector<double> temp(gradients);
      temp.Scale(1.0/tot_t_weighted);
      if (print_avg_gradients) {
        KALDI_LOG << "Vector of average gradients wrt output activations is: \n" << temp;
      } else {
        KALDI_VLOG(4) << "Vector of average gradients wrt output activations is: \n" << temp;
      }
    }
  }
  if (AccumulateOutput()) {
    {
      Vector<double> temp(output);
      temp.Scale(1.0/tot_t_weighted);
      if (print_avg_output) {
        KALDI_LOG << "Average DNN posterior is: \n" << temp;
      } else {
        KALDI_VLOG(4) << "Average DNN posterior is: \n" << temp;
      }
    }
  }

  if (AccumulateCounts()) {
    {
      {
        Vector<double> temp(indication_counts);
        temp.Scale(1.0/tot_t_weighted);
        if (print_avg_counts) {
          KALDI_LOG << "Average indication counts is: \n" << temp;
        } else {
          KALDI_VLOG(4) << "Average indication counts is: \n" << temp;
        }
      }
    }
  }
}

void DiscriminativeTrainingStats::PrintAvgGradientForPdf(int32 pdf_id) const {
  if (AccumulateCounts()) {
    if (pdf_id < gradients.Dim() and pdf_id >= 0) {
      KALDI_LOG << "Average gradient wrt output activations of pdf " << pdf_id 
                << " is " << gradients(pdf_id) / tot_t_weighted
                << " per frame, over "
                << tot_t_weighted << " frames";
    } 
  }
}



}  // namespace discriminative
}  // namespace kaldi

