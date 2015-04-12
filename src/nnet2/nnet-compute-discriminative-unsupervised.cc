// nnet2/nnet-compute-discriminative-unsupervised.cc

// Copyright 2014   Vimal Manohar

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

#include "nnet2/nnet-compute-discriminative-unsupervised.h"
#include "lat/lattice-functions.h"
#include "base/kaldi-types-extra.h"

namespace kaldi {
namespace nnet2 {

typedef SignedLogReal<double> SignedLogDouble;

NnetDiscriminativeUnsupervisedUpdater::NnetDiscriminativeUnsupervisedUpdater(
    const AmNnet &am_nnet,
    const TransitionModel &tmodel,
    const NnetDiscriminativeUnsupervisedUpdateOptions &opts,
    const DiscriminativeUnsupervisedNnetExample &eg,
    Nnet *nnet_to_update,
    NnetDiscriminativeUnsupervisedStats *stats,
    std::vector<BaseFloat> *weights):
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


    
SubMatrix<BaseFloat> NnetDiscriminativeUnsupervisedUpdater::GetInputFeatures() const {
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

void NnetDiscriminativeUnsupervisedUpdater::Propagate() {
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

    component.Propagate(chunk_info_out_[c], chunk_info_out_[c+1], input, &output);
    const Component *prev_component = (c == 0 ? NULL : &(nnet.GetComponent(c-1)));
    bool will_do_backprop = (nnet_to_update_ != NULL),
        keep_last_output = will_do_backprop &&
        ((c>0 && prev_component->BackpropNeedsOutput()) ||
         component.BackpropNeedsInput());
    if (!keep_last_output)
      forward_data_[c].Resize(0, 0); // We won't need this data; save memory.
  }
}



SignedLogDouble NnetDiscriminativeUnsupervisedUpdater::LatticeComputations() {
  ConvertLattice(eg_.lat, &lat_); // convert to Lattice.
  TopSort(&lat_); // Topologically sort (required by forward-backward algorithms)

  std::vector<int32> state_times;
  int32 T = LatticeStateTimes(lat_, &state_times);
  
  if (stats_ != NULL) {
    stats_->tot_t += T;
    stats_->tot_t_weighted += T * eg_.weight;
  }

  const VectorBase<BaseFloat> &priors = am_nnet_.Priors();        
  const CuMatrix<BaseFloat> &posteriors = forward_data_.back();   // Acoustic posteriors

  KALDI_ASSERT(posteriors.NumRows() == T);
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
  requested_indexes.reserve(wiggle_room * lat_.NumStates());

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

  std::vector<BaseFloat> answers;
  posteriors.Lookup(requested_indexes, &answers);
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
    BaseFloat pseudo_loglike = log(post / priors(pdf_id)) * opts_.acoustic_scale;
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

  // Get the NCE objective function derivatives wrt to the activations
  Posterior post;

  //CuMatrixBase<BaseFloat> &output(GetOutput());
  int32 num_components = am_nnet_.GetNnet().NumComponents();
  const CuMatrix<BaseFloat> &output(forward_data_[num_components]);
  backward_data_.Resize(output.NumRows(), output.NumCols()); // zeroes it.

  NnetDiscriminativeUnsupervisedStats this_stats(output.NumCols());
  if (stats_ == NULL || !stats_->store_gradients) {
    this_stats.store_gradients = false;
  }

  SignedLogDouble objf = GetDerivativesWrtActivations(&post);
  this_stats.tot_objf += eg_.weight * objf.Value();
  
  // Scale the derivatives by the weight. 
  // Equivalent to having different learning rates for different egs
  ScalePosterior(eg_.weight, &post);

  KALDI_ASSERT(output.NumRows() == post.size());

  double tot_post = 0.0;
  std::vector<MatrixElement<BaseFloat> > sv_labels;
  sv_labels.reserve(answers.size());
  for (int32 t = 0; t < post.size(); t++) {
    for (int32 i = 0; i < post[t].size(); i++) {
      int32 pdf_id = post[t][i].first;
      BaseFloat weight = post[t][i].second;
      MatrixElement<BaseFloat> elem = {t, pdf_id, weight};
      sv_labels.push_back(elem);
      tot_post += (weight > 0 ? weight : -weight);
    }
  }

  this_stats.tot_gradients += tot_post;

  { // We don't actually need tot_objf and tot_weight; we have already
    // computed the objective function.
    BaseFloat tot_objf, tot_weight;
    backward_data_.CompObjfAndDeriv(sv_labels, output, &tot_objf, &tot_weight);
    // Now backward_data_ will contan the derivative at the output.
    // Our work here is done..
    if (this_stats.store_gradients)
      (this_stats.gradients).AddRowSumMat(1.0, CuMatrix<double>(backward_data_));
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


SignedLogDouble NnetDiscriminativeUnsupervisedUpdater::GetDerivativesWrtActivations(Posterior *post) {
  Posterior tid_post;
  SignedLogDouble obj_func;

  if (opts_.criterion == "nce") {
    if (opts_.boost != 0.0) {
      //BaseFloat max_silence_error = 0.0;
      // KALDI_ASSERT(post_.size() > 0);
      KALDI_ERR << "Boost is not currently supported!";
      //obj_func = LatticeForwardBackwardNceBoosted(tmodel_, post_,
      //    silence_phones_, opts_.boost,
      //    max_silence_error, lat_, &tid_post);
    } else {
      if (eg_.weights.size() > 0)
        obj_func = LatticeForwardBackwardNce(tmodel_, lat_, &tid_post, &eg_.weights, opts_.weight_threshold);
      else
        obj_func = LatticeForwardBackwardNce(tmodel_, lat_, &tid_post);
    }
  } else if (opts_.criterion == "esmbr") {
    obj_func = static_cast<SignedLogDouble>(
        LatticeForwardBackwardEmpeVariants(tmodel_, 
        silence_phones_, lat_, opts_.criterion,
        opts_.one_silence_class, 
        &tid_post, opts_.weight_threshold));
  } else if (opts_.criterion == "smbr") {
    obj_func = static_cast<SignedLogDouble>(
        LatticeForwardBackwardMpeVariants(tmodel_, 
        silence_phones_, lat_, eg_.ali,
        opts_.criterion,
        opts_.one_silence_class, 
        &tid_post, &eg_.weights, opts_.weight_threshold));
  }
  
  ConvertPosteriorToPdfs(tmodel_, tid_post, post);

  if ((*post)[0].size() == 0)
    KALDI_WARN << "0 size posterior";

  int32 phone_acc = 0, pdf_acc = 0, best_path_phone_acc = 0, 
        best_path_pdf_acc = 0, 
        best_path_phone_match = 0, best_path_pdf_match = 0;
  
  if (GetVerboseLevel() > 3) {
    for (int32 t = 0; t < tid_post.size(); t++) {
      int32 phone = -1, ali_phone = -1;
      int32 pdf = -1, ali_pdf = -1;
      BaseFloat weight = -9e30;
      for (int32 j = 0; j < tid_post[t].size(); j++) {
        if (tid_post[t][j].second > weight) {
          weight = tid_post[t][j].second;
          phone = tmodel_.TransitionIdToPhone(tid_post[t][j].first);
          pdf = tmodel_.TransitionIdToPdf(tid_post[t][j].first);
        }
      }
      
      if (post_.size() > 0) {
        ali_phone = tmodel_.TransitionIdToPhone(post_[t][0].first);
        ali_pdf = tmodel_.TransitionIdToPdf(post_[t][0].first);

        if (phone == ali_phone) 
          best_path_phone_match++;
        if (pdf == ali_pdf)
          best_path_pdf_match++;
      }

      if (eg_.oracle_ali.size() > 0) {
        int32 oracle_ali_phone = tmodel_.TransitionIdToPhone(eg_.oracle_ali[t]);
        int32 oracle_ali_pdf = tmodel_.TransitionIdToPdf(eg_.oracle_ali[t]);
        if (phone == oracle_ali_phone) {
          phone_acc++;
        }
        if (pdf == oracle_ali_pdf) {
          pdf_acc++;
        }
        
        if (ali_phone == oracle_ali_phone) {
          best_path_phone_acc++;
        }
        if (ali_pdf == oracle_ali_pdf) {
          best_path_pdf_acc++;
        }
      }
    }
    KALDI_LOG << "Phone accuracy is " << phone_acc 
      << "; pdf accuracy is " << pdf_acc 
      << " over " << tid_post.size()  << " frames.";
    KALDI_LOG << "Best path phone accuracy is " << best_path_phone_acc 
      << "; Best path pdf accuracy is " << best_path_pdf_acc 
      << " over " << tid_post.size()  << " frames.";
    KALDI_LOG << "Best path phone match is " << best_path_phone_match 
      << "; Best path pdf match is " << best_path_pdf_match
      << " over " << tid_post.size()  << " frames.";
  }

  if (eg_.oracle_ali.size() == 0 && GetVerboseLevel() > 5) {
    KALDI_LOG << "Printing phone confusions in lattice and the resulting gradients: Frame Confusion <time> <phone> <hyp-phone> <pdf> <hyp-pdf>";
    for (int32 t = 0; t < tid_post.size(); t++) {
      for (int32 j = 0; j < tid_post[t].size(); j++) {
        int32 phone = tmodel_.TransitionIdToPhone(tid_post[t][j].first);
        int32 pdf = tmodel_.TransitionIdToPdf(tid_post[t][j].first);
        if (post_.size() > 0) {
          int32 ali_phone = tmodel_.TransitionIdToPhone(post_[t][0].first);
          int32 ali_pdf = tmodel_.TransitionIdToPdf(post_[t][0].first);
          KALDI_LOG << "PhoneConfusion: " << t << " " 
                    << phone << " " << ali_phone << " "
                    << pdf << " " << ali_pdf << " "
                    << tid_post[t][j].second;
        } else {
          KALDI_LOG << "PhoneConfusion: " << t << " " 
                    << phone << " " 
                    << pdf << " "
                    << tid_post[t][j].second;
        }
      }
    }
  } else if ( GetVerboseLevel() > 5) {
    KALDI_LOG << "Printing phone confusions in lattice and the resulting gradients: Frame Confusion <time> <phone> <ref-phone> [<hyp-phone>] <pdf> <ref-pdf> [<hyp-pdf>]";
    for (int32 t = 0; t < tid_post.size(); t++) {
      for (int32 j = 0; j < tid_post[t].size(); j++) {
        int32 phone = tmodel_.TransitionIdToPhone(tid_post[t][j].first);
        int32 pdf = tmodel_.TransitionIdToPdf(tid_post[t][j].first);
        int32 oracle_ali_phone = tmodel_.TransitionIdToPhone(eg_.oracle_ali[t]);
        int32 oracle_ali_pdf = tmodel_.TransitionIdToPdf(eg_.oracle_ali[t]);
        if (post_.size() > 0) {
          int32 ali_phone = tmodel_.TransitionIdToPhone(post_[t][0].first);
          int32 ali_pdf = tmodel_.TransitionIdToPdf(post_[t][0].first);
          KALDI_LOG << "PhoneConfusion: " << t << " " 
                    << phone << " " << oracle_ali_phone << " " << ali_phone << " " 
                    << pdf << " " << oracle_ali_pdf << " " << ali_pdf << " "
                    << tid_post[t][j].second;
        } else {
          KALDI_LOG << "PhoneConfusion: " << t << " " 
                    << phone << " " << oracle_ali_phone << " "
                    << pdf << " " << oracle_ali_pdf << " "
                    << tid_post[t][j].second;
        }
      }
    }
  }

  return obj_func;
}

void NnetDiscriminativeUnsupervisedUpdater::Backprop() {
  const Nnet &nnet = am_nnet_.GetNnet();
  for (int32 c = nnet.NumComponents() - 1; c >= 0; c--) {
    const Component &component = nnet.GetComponent(c);
    Component *component_to_update = &(nnet_to_update_->GetComponent(c));
    const CuMatrix<BaseFloat>  &input = forward_data_[c],
                            &output = forward_data_[c+1],
                      &output_deriv = backward_data_;
    CuMatrix<BaseFloat> input_deriv;
    component.Backprop(chunk_info_out_[c], chunk_info_out_[c+1], input, output, 
        output_deriv, component_to_update, &input_deriv);
    backward_data_.Swap(&input_deriv); // backward_data_ = input_deriv.
  }
}

SignedLogDouble NnetDiscriminativeUnsupervisedUpdate(const AmNnet &am_nnet,
                              const TransitionModel &tmodel,
                              const NnetDiscriminativeUnsupervisedUpdateOptions &opts,
                              const DiscriminativeUnsupervisedNnetExample &eg,
                              Nnet *nnet_to_update,
                              NnetDiscriminativeUnsupervisedStats *stats) {
  NnetDiscriminativeUnsupervisedUpdater updater(am_nnet, tmodel, opts, eg,
                                                nnet_to_update, stats);
  SignedLogDouble objf = updater.Update();
  return objf;
}

void NnetDiscriminativeUnsupervisedStats::Add(const NnetDiscriminativeUnsupervisedStats &other) {
  tot_t += other.tot_t;
  tot_t_weighted += other.tot_t_weighted;
  tot_objf += other.tot_objf;
  tot_gradients += other.tot_gradients;

  if (store_gradients) {
    gradients.AddVec(1.0, other.gradients);
  }
}

void NnetDiscriminativeUnsupervisedStats::Print(string criterion) const {
  double objf = tot_objf / tot_t_weighted;
  double avg_gradients = tot_gradients / tot_t_weighted;

  if (criterion == "nce") {
    KALDI_LOG << "Average modulus of NCE gradients is " << avg_gradients 
              << " per frame, over "
              << tot_t_weighted << " frames";
    KALDI_LOG << "NCE objective function is " << objf << " per frame, over "
              << tot_t_weighted << " frames";
  } else {
    KALDI_LOG << "Average modulus of ESMBR gradients is " << avg_gradients 
              << " per frame, over "
              << tot_t_weighted << " frames";
    KALDI_LOG << "ESMBR objective function is " << objf << " per frame, over "
              << tot_t_weighted << " frames";
  }

  if (store_gradients) {
    Vector<double> temp(gradients);
    temp.Scale(1.0/tot_t_weighted);
    KALDI_VLOG(4) << "Vector of average gradients wrt output activations is: \n" << temp;
  }
}

void NnetDiscriminativeUnsupervisedStats::PrintPost(int32 pdf_id) const {
  if (store_gradients) {
    if (pdf_id < gradients.Dim() and pdf_id >= 0) {
      KALDI_LOG << "Average posterior of pdf " << pdf_id 
                << " is " << gradients(pdf_id) / tot_t_weighted
                << " per frame, over "
                << tot_t_weighted << " frames";
    } 
  }
}

} // namespace nnet2
} // namespace kaldi
            
