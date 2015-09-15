// nnet2/nnet-compute-discriminative.cc

// Copyright 2012-2013   Johns Hopkins University (author: Daniel Povey)
//           2014-2015   Vimal Manohar

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

#include "nnet2/nnet-compute-discriminative.h"
#include "hmm/posterior.h"
#include "lat/lattice-functions.h"
#include "base/kaldi-types-extra.h"

namespace kaldi {
namespace nnet2 {

typedef SignedLogReal<double> SignedLogDouble;

NnetDiscriminativeUpdater::NnetDiscriminativeUpdater(
    const AmNnet &am_nnet,
    const TransitionModel &tmodel,
    const NnetDiscriminativeUpdateOptions &opts,
    const DiscriminativeNnetExample &eg,
    Nnet *nnet_to_update,
    NnetDiscriminativeStats *stats):
    am_nnet_(am_nnet), tmodel_(tmodel), opts_(opts), eg_(eg),
    nnet_to_update_(nnet_to_update), stats_(stats) {
  if (!SplitStringToIntegers(opts_.silence_phones_str, ":", false,
                             &silence_phones_)) {
    KALDI_ERR << "Bad value for --silence-phones option: "
              << opts_.silence_phones_str;
  }
  const Nnet &nnet = am_nnet_.GetNnet();
  nnet.ComputeChunkInfo(eg_.input_frames.NumRows(), 1, &chunk_info_out_);
}



SubMatrix<BaseFloat> NnetDiscriminativeUpdater::GetInputFeatures() const {
  int32 num_frames_output = eg_.num_frames;
  int32 eg_left_context = eg_.left_context,
      eg_right_context = eg_.input_frames.NumRows() -
      num_frames_output - eg_left_context;
  KALDI_ASSERT(eg_right_context >= 0);
  const Nnet &nnet = am_nnet_.GetNnet();
  // Make sure the example has enough acoustic left and right
  // context... normally we'll use examples generated using the same model,
  // which will have the exact context, but we enable a mismatch in context as
  // long as it is more, not less.
  KALDI_ASSERT(eg_left_context >= nnet.LeftContext() &&
               eg_right_context >= nnet.RightContext());
  int32 offset = eg_left_context - nnet.LeftContext(),
      num_output_frames =
      num_frames_output + nnet.LeftContext() + nnet.RightContext();
  SubMatrix<BaseFloat> ans(eg_.input_frames, offset, num_output_frames,
                           0, eg_.input_frames.NumCols());
  return ans;
}

void NnetDiscriminativeUpdater::Propagate() {
  const Nnet &nnet = am_nnet_.GetNnet();
  forward_data_.resize(nnet.NumComponents() + 1);

  SubMatrix<BaseFloat> input_feats = GetInputFeatures();
  int32 spk_dim = eg_.spk_info.Dim();
  if (spk_dim == 0) {
    forward_data_[0] = input_feats;
  } else {
    // If there is speaker vector, then copy it to the last columns in
    // all the rows
    forward_data_[0].Resize(input_feats.NumRows(),
                            input_feats.NumCols() + eg_.spk_info.Dim());
    forward_data_[0].Range(0, input_feats.NumRows(),
                           0, input_feats.NumCols()).CopyFromMat(input_feats);
    forward_data_[0].Range(0, input_feats.NumRows(),
                           input_feats.NumCols(), spk_dim).CopyRowsFromVec(
                               eg_.spk_info);
  }

  for (int32 c = 0; c < nnet.NumComponents(); c++) {
    const Component &component = nnet.GetComponent(c);
    CuMatrix<BaseFloat> &input = forward_data_[c],
        &output = forward_data_[c+1];
    component.Propagate(chunk_info_out_[c] , chunk_info_out_[c+1], input, &output);
    const Component *prev_component = (c == 0 ? NULL :
                                       &(nnet.GetComponent(c-1)));
    bool will_do_backprop = (nnet_to_update_ != NULL),
        keep_last_output = will_do_backprop &&
        ((c>0 && prev_component->BackpropNeedsOutput()) ||
         component.BackpropNeedsInput());
    
    if (nnet_to_update_ != NULL) {
      if (stats_->store_logit_stats && 
          (dynamic_cast<SoftmaxComponent*>(&(nnet_to_update_->GetComponent(c)))) != NULL) {
        if (stats_->logit.Dim() == 0) {
          (stats_->logit).Resize(input.NumCols());
          (stats_->logit_gradients).Resize(input.NumCols());
        }
        (stats_->logit).AddRowSumMat(1.0, CuMatrix<double>(input));
      }
    }

    if (!keep_last_output)
      forward_data_[c].Resize(0, 0); // We won't need this data; save memory.
  }
}



SignedLogDouble NnetDiscriminativeUpdater::LatticeComputations() {
  ConvertLattice(eg_.den_lat, &lat_); // convert to Lattice.

  if ((opts_.criterion == "esmbr" || opts_.criterion == "empfe" || opts_.criterion == "nce" ) && eg_.num_lat_present) {
    ConvertLattice(eg_.num_lat, &num_lat_);
    TopSort(&num_lat_); // Topologically sort (required by forward-backward algorithms)
  }

  TopSort(&lat_); // Topologically sort (required by forward-backward algorithms)

  if (opts_.criterion == "mmi" && opts_.boost != 0.0) {
    BaseFloat max_silence_error = 0.0;
    LatticeBoost(tmodel_, eg_.num_ali, silence_phones_,
                 opts_.boost, max_silence_error, &lat_);
  }
  
  int32 num_frames = eg_.num_frames;

  if (stats_ != NULL) {
    stats_->tot_t += num_frames;
    stats_->tot_t_weighted += num_frames * eg_.weight;
  }
  
  const VectorBase<BaseFloat> &priors = am_nnet_.Priors();
  const CuMatrix<BaseFloat> &posteriors = forward_data_.back();

  KALDI_ASSERT(posteriors.NumRows() == num_frames);
  int32 num_pdfs = posteriors.NumCols();
  KALDI_ASSERT(num_pdfs == priors.Dim());
  
  // We need to look up the posteriors of some pdf-ids in the matrix
  // "posteriors".  Rather than looking them all up using operator (), which is
  // very slow because each lookup involves a separate CUDA call with
  // communication over PciExpress, we look them up all at once using
  // CuMatrix::Lookup().
  // Note: regardless of the criterion, we evaluate the likelihoods in
  // the numerator alignment.  Even though they may be irrelevant to
  // the optimization, they will affect the value of the objective function.
  
  std::vector<Int32Pair> requested_indexes;
  BaseFloat wiggle_room = 1.3; // value not critical.. it's just 'reserve'

  int32 num_reserve = wiggle_room * lat_.NumStates();
  
  if (opts_.criterion == "mmi") {
    // For looking up the posteriors corresponding to the pdfs in the alignment
    num_reserve += num_frames;
  } else if (opts_.criterion == "nce" || opts_.criterion == "empfe" || 
             opts_.criterion == "esmbr") {
    // For looking up the posteriors corresponding to the pdfs in the 
    // numerator lattice or the numerator posteriors
    if (eg_.num_lat_present) num_reserve *= 2;
    else if (eg_.num_post.size() > 0) 
      num_reserve += 2 * wiggle_room * num_frames;
  }
  
  requested_indexes.reserve(num_reserve);
  
  // Denominator probabilities to look up from denominator lattice
  std::vector<int32> state_times;
  int32 T = LatticeStateTimes(lat_, &state_times);
  KALDI_ASSERT(T == num_frames);
  
  StateId num_states = lat_.NumStates();
  for (StateId s = 0; s < num_states; s++) {
    StateId t = state_times[s];
    for (fst::ArcIterator<Lattice> aiter(lat_, s); !aiter.Done(); aiter.Next()) {
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
      int32 tid = eg_.num_ali[t], pdf_id = tmodel_.TransitionIdToPdf(tid);
      KALDI_ASSERT(pdf_id >= 0 && pdf_id < num_pdfs);
      requested_indexes.push_back(MakePair(t, pdf_id));
    }
  } else if (opts_.criterion == "nce" || opts_.criterion == "empfe" || opts_.criterion == "esmbr") {
    if (eg_.num_lat_present) {
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
  if (num_floored > 0) {
    KALDI_WARN << "Floored " << num_floored << " probabilities from nnet.";
  }
  
  index = 0;

  // Now put the negative (scaled) acoustic log-likelihoods in the lattice.
  for (StateId s = 0; s < num_states; s++) {
    for (fst::MutableArcIterator<Lattice> aiter(&lat_, s);
         !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel != 0) { // input-side has transition-ids, output-side empty
        arc.weight.SetValue2(-answers[index]);
        index++;
        aiter.SetValue(arc);
      }
    }
    LatticeWeight final = lat_.Final(s);
    if (final != LatticeWeight::Zero()) {
      final.SetValue2(0.0); // make sure no acoustic term in final-prob.
      lat_.SetFinal(s, final);
    }
  }
  
  // Look up numerator probabilities corresponding to alignment
  if (opts_.criterion == "mmi") {
    double tot_num_like = 0.0;
    for (; index < eg_.num_ali.size(); index++)
      tot_num_like += answers[index];
    stats_->tot_num_objf += eg_.weight * tot_num_like;
  } else if (opts_.criterion == "nce" || opts_.criterion == "empfe" || opts_.criterion == "esmbr") {
    if (eg_.num_lat_present) {
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
  
  int32 num_components = am_nnet_.GetNnet().NumComponents();
  const CuMatrix<BaseFloat> &output(forward_data_[num_components]);
  backward_data_.Resize(output.NumRows(), output.NumCols()); // zeroes it.

  NnetDiscriminativeStats this_stats(output.NumCols());
  if (stats_ == NULL || !stats_->store_gradients) {
    this_stats.store_gradients = false;
  }
  
  Posterior post;

  // Can be den_objf for mmi
  SignedLogDouble objf = GetDerivativesWrtActivations(&post);
  this_stats.tot_objf += eg_.weight * objf.Value();

  ScalePosterior(eg_.weight, &post);

  KALDI_ASSERT(output.NumRows() == post.size());
  
  double tot_num_post = 0.0, tot_post = 0.0, tot_den_post = 0.0;
  std::vector<MatrixElement<BaseFloat> > sv_labels;
  sv_labels.reserve(answers.size());
  for (int32 t = 0; t < post.size(); t++) {
    for (int32 i = 0; i < post[t].size(); i++) {
      int32 pdf_id = post[t][i].first;
      this_stats.indication_counts(pdf_id) += 1.0;
      BaseFloat weight = post[t][i].second;
      MatrixElement<BaseFloat> elem = {t, pdf_id, weight};
      sv_labels.push_back(elem);
      if (opts_.criterion != "nce") {
        if (weight > 0.0) { tot_num_post += weight; }
        else { tot_den_post -= weight; }
      } else {
        tot_post += (weight > 0.0 ? weight: - weight);
      }
    }
  }

  this_stats.tot_gradients += tot_post;
  this_stats.tot_den_count += tot_den_post;
  this_stats.tot_num_count += tot_num_post;
  
  { // We don't actually need tot_objf and tot_weight; we have already
    // computed the objective function.
    BaseFloat tot_objf, tot_weight;
    backward_data_.CompObjfAndDeriv(sv_labels, output, &tot_objf, &tot_weight);
    // Now backward_data_ will contan the derivative at the output.
    // Our work here is done..
    if (this_stats.store_gradients) {
      (this_stats.gradients).AddRowSumMat(1.0, CuMatrix<double>(backward_data_));
      (this_stats.output).AddRowSumMat(1.0, CuMatrix<double>(output));
    }
  }
  
  if (stats_ != NULL)
    stats_->Add(this_stats);

  // For the purpose of printing this_stats
  this_stats.tot_t = T;
  this_stats.tot_t_weighted = T * eg_.weight;

  if (GetVerboseLevel() >= 4) {
    this_stats.Print(opts_.criterion);
  }

  // Now backward_data_ will contan the derivative at the output.
  // Our work here is done..
  return objf;
}


SignedLogDouble NnetDiscriminativeUpdater::GetDerivativesWrtActivations(Posterior *post) {
  if (opts_.criterion == "mpfe" || opts_.criterion == "smbr") {
    Posterior tid_post;
    double ans = LatticeForwardBackwardMpeVariants(tmodel_, silence_phones_, lat_,
        eg_.num_ali, opts_.criterion,
        opts_.one_silence_class,
        &tid_post);
    ConvertPosteriorToPdfs(tmodel_, tid_post, post);
    return static_cast<SignedLogDouble>(ans); // returns the objective function.
  } else if (opts_.criterion == "mmi") {
    bool convert_to_pdfs = true, cancel = true;
    // we'll return the denominator-lattice forward backward likelihood,
    // which is one term in the objective function.
    return static_cast<SignedLogDouble>(LatticeForwardBackwardMmi(tmodel_, lat_, eg_.num_ali,
        opts_.drop_frames, convert_to_pdfs,
        cancel, post));
  } else if (opts_.criterion == "nce") {
    Posterior tid_post;

    SignedLogDouble obj_func;

    if (eg_.weights.size() > 0)
      obj_func = LatticeForwardBackwardNce(tmodel_, lat_, &tid_post, &eg_.weights, opts_.weight_threshold);
    else
      obj_func = LatticeForwardBackwardNce(tmodel_, lat_, &tid_post);

    ConvertPosteriorToPdfs(tmodel_, tid_post, post);
    return obj_func; // returns the objective function.
  } else if (opts_.criterion == "empfe" || opts_.criterion == "esmbr") {
    Posterior tid_post;

    SignedLogDouble obj_func;

    if (!eg_.num_lat_present && eg_.num_post.size() > 0) {
      // Using numerator posteriors
      obj_func = static_cast<SignedLogDouble>(
          LatticeForwardBackwardEmpeVariants(tmodel_, 
            silence_phones_, lat_, eg_.num_ali, &eg_.num_post, NULL,
            opts_.criterion, opts_.one_silence_class, opts_.deletion_penalty, 
            &tid_post, opts_.weight_threshold));
    } else if (eg_.num_lat_present) {
      // Using numerator lattice
      obj_func = static_cast<SignedLogDouble>(
          LatticeForwardBackwardEmpeVariants(tmodel_, 
            silence_phones_, lat_, eg_.num_ali, NULL, &num_lat_, 
            opts_.criterion, opts_.one_silence_class, opts_.deletion_penalty, 
            &tid_post, opts_.weight_threshold));
    } else {
      // Using denominator lattice
      obj_func = static_cast<SignedLogDouble>(
          LatticeForwardBackwardEmpeVariants(tmodel_, 
            silence_phones_, lat_, eg_.num_ali, NULL, NULL,
            opts_.criterion, opts_.one_silence_class, opts_.deletion_penalty, 
            &tid_post, opts_.weight_threshold));
    }

    ConvertPosteriorToPdfs(tmodel_, tid_post, post);
    return obj_func;
  }

  return SignedLogDouble(0.0);
}


void NnetDiscriminativeUpdater::Backprop() {
  const Nnet &nnet = am_nnet_.GetNnet();
  for (int32 c = nnet.NumComponents() - 1; c >= 0; c--) {
    const Component &component = nnet.GetComponent(c);
    Component *component_to_update = &(nnet_to_update_->GetComponent(c));
    const CuMatrix<BaseFloat>  &input = forward_data_[c],
                            &output = forward_data_[c+1],
                      &output_deriv = backward_data_;
    CuMatrix<BaseFloat> input_deriv;
    component.Backprop(chunk_info_out_[c], chunk_info_out_[c+1], input, output, output_deriv,
                       component_to_update, &input_deriv);
    backward_data_.Swap(&input_deriv); // backward_data_ = input_deriv.
  }
}


SignedLogDouble NnetDiscriminativeUpdate(const AmNnet &am_nnet,
                        const TransitionModel &tmodel,
                        const NnetDiscriminativeUpdateOptions &opts,
                        const DiscriminativeNnetExample &eg,
                        Nnet *nnet_to_update,
                        NnetDiscriminativeStats *stats) {
  NnetDiscriminativeUpdater updater(am_nnet, tmodel, opts, eg,
                              nnet_to_update, stats);
  SignedLogDouble objf = updater.Update();
  return objf;
}

void NnetDiscriminativeStats::Add(const NnetDiscriminativeStats &other) {
  tot_t += other.tot_t;
  tot_t_weighted += other.tot_t_weighted;
  tot_objf += other.tot_objf;             // Actually tot_den_objf for mmi
  tot_gradients += other.tot_gradients;   // Only for nce 
  tot_num_count += other.tot_num_count;   // Not for nce
  tot_den_count += other.tot_den_count;   // Not for nce
  tot_num_objf += other.tot_num_objf;     // Only for mmi
  
  if (store_gradients) {
    gradients.AddVec(1.0, other.gradients);
    output.AddVec(1.0, other.output);
    indication_counts.AddVec(1.0, other.indication_counts);
    if (logit.Dim() == 0 && other.logit.Dim() > 0) {
      logit.Resize(other.logit.Dim());
      logit_gradients.Resize(other.logit.Dim());
      logit.AddVec(1.0, other.logit);
      logit_gradients.AddVec(1.0, other.logit_gradients);
    }
  }
}

void NnetDiscriminativeStats::Print(std::string criterion, bool print_gradients, bool print_post) const {
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
  
  if (store_gradients) {
    {
      Vector<double> temp(gradients);
      temp.Scale(1.0/tot_t_weighted);
      if (print_post) {
        KALDI_LOG << "Vector of average gradients wrt output activations is: \n" << temp;
      } else {
        KALDI_VLOG(4) << "Vector of average gradients wrt output activations is: \n" << temp;
      }
    }
    {
      Vector<double> temp(output);
      temp.Scale(1.0/tot_t_weighted);
      if (print_post) {
        KALDI_LOG << "Average DNN posterior is: \n" << temp;
      } else {
        KALDI_VLOG(4) << "Average DNN posterior is: \n" << temp;
      }
    }
  }

  if (store_logit_stats) {
    {
      Vector<double> temp(logit_gradients);
      temp.Scale(1.0/tot_t_weighted);
      if (print_gradients) {
        KALDI_LOG << "Vector of average gradients wrt logits is: \n" << temp;
      } else {
        KALDI_VLOG(4) << "Vector of average gradients wrt logits: \n" << temp;
      }
    }
    {
      Vector<double> temp(logit);
      temp.Scale(1.0/tot_t_weighted);
      if (print_post) {
        KALDI_LOG << "Average logit is: \n" << temp;
      } else {
        KALDI_VLOG(4) << "Average logit is: \n" << temp;
      }
    }
    {
      {
        Vector<double> temp(indication_counts);
        temp.Scale(1.0/tot_t_weighted);
        if (print_post) {
          KALDI_LOG << "Average indication counts is: \n" << temp;
        } else {
          KALDI_VLOG(4) << "Average indication counts is: \n" << temp;
        }
      }
    }
  }
}

void NnetDiscriminativeStats::PrintPost(int32 pdf_id) const {
  if (store_gradients) {
    if (pdf_id < gradients.Dim() and pdf_id >= 0) {
      KALDI_LOG << "Average gradient wrt output activations of pdf " << pdf_id 
                << " is " << gradients(pdf_id) / tot_t_weighted
                << " per frame, over "
                << tot_t_weighted << " frames";
    } 
  }
}

} // namespace nnet2
} // namespace kaldi
