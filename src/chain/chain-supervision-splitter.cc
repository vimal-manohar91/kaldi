// chain/chain-supervision-splitter.cc

// Copyright 2012-2015  Johns Hopkins University (author: Daniel Povey)
//           2014-2015  Vimal Manohar
//                2017  Vimal Manohar

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

#include "chain/chain-supervision-splitter.h"
#include "chain/chain-supervision.h"
#include "lat/lattice-functions.h"

namespace kaldi {
namespace chain {

typedef fst::ArcTpl<LatticeWeight> LatticeArc;
typedef fst::VectorFst<LatticeArc> Lattice;

void FstToLattice(const fst::StdVectorFst &fst, Lattice *lat) {
  lat->DeleteStates();

  int32 start_state = fst.Start();
  for (int32 i = 0; i < fst.NumStates(); i++)
    lat->AddState();

  lat->SetStart(start_state);

  for (fst::StdArc::StateId s = 0; s < fst.NumStates(); s++) {
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, s);
          !aiter.Done(); aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();

      LatticeWeight weight = LatticeWeight::One();
      weight.SetValue1(arc.weight.Value());

      lat->AddArc(s, 
                  LatticeArc(arc.ilabel, arc.olabel, weight, arc.nextstate));
    }

    if (fst.Final(s) != fst::TropicalWeight::Zero()) {
      LatticeWeight weight = LatticeWeight::One();
      weight.SetValue1(fst.Final(s).Value());
      lat->SetFinal(s, weight);
    }
  }
}

/** This function converts lattice to FSA with weight equal to
    sum of acoustic and language score, and pdf_id + 1 as labels. 
    This assumes that the acoustic and language scores are scaled appropriately.
*/
void ConvertLatticeToPdfLabels(
    const TransitionModel &tmodel,
    const Lattice &ifst,
    fst::StdVectorFst *ofst) {
  typedef fst::ArcTpl<LatticeWeight> ArcIn;
  typedef fst::StdArc ArcOut;
  typedef ArcIn::StateId StateId;
  ofst->DeleteStates();
  // The states will be numbered exactly the same as the original FST.
  // Add the states to the new FST.
  StateId num_states = ifst.NumStates();
  for (StateId s = 0; s < num_states; s++)
    ofst->AddState();
  ofst->SetStart(ifst.Start());
  for (StateId s = 0; s < num_states; s++) {
    LatticeWeight final_iweight = ifst.Final(s);
    if (final_iweight != LatticeWeight::Zero()) {
      fst::TropicalWeight final_oweight;
      ConvertLatticeWeight(final_iweight, &final_oweight);
      ofst->SetFinal(s, final_oweight);
    }
    for (fst::ArcIterator<Lattice> iter(ifst, s);
         !iter.Done();
         iter.Next()) {
      const ArcIn &arc = iter.Value();
      KALDI_PARANOID_ASSERT(arc.weight != LatticeWeight::Zero());
      ArcOut oarc;
      ConvertLatticeWeight(arc.weight, &oarc.weight);
      if (arc.ilabel == 0)
        oarc.ilabel = 0;  // epsilon arc
      else
        oarc.ilabel = tmodel.TransitionIdToPdf(arc.ilabel) + 1;  // pdf + 1
      oarc.olabel = oarc.ilabel;
      oarc.nextstate = arc.nextstate;
      ofst->AddArc(s, oarc);
    }
  }
}

bool LatticeToNumeratorPost(const Lattice &lat,
                            const TransitionModel &trans_model,
                            const fst::StdVectorFst &fst,
                            Posterior *post, std::string key) {
  fst::StdVectorFst sup_fst;
  ConvertLatticeToPdfLabels(trans_model, lat, &sup_fst);

  if (!AddWeightToFst(fst, &sup_fst)) {
    if (!key.empty())
      KALDI_WARN << "For key " << key << ", ";
    KALDI_WARN << "FST was empty after composing with FST. "
               << "This should be extremely rare (a few per corpus, at most)";
    return false;
  }

  // Convert fst to lattice to extract posterior using forward backward.
  Lattice lat_copy;
  ConvertFstToLattice(sup_fst, &lat_copy);

  kaldi::uint64 props = lat_copy.Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted)) {
    if (fst::TopSort(&lat_copy) == false)
      KALDI_ERR << "Cycles detected in lattice.";
  }

  LatticeForwardBackward(lat_copy, post);
  return true;
}

SupervisionLatticeSplitter::SupervisionLatticeSplitter(
    const SupervisionLatticeSplitterOptions &opts,
    const SupervisionOptions &sup_opts,
    const TransitionModel &trans_model):
  sup_opts_(sup_opts), opts_(opts), trans_model_(trans_model), 
  incomplete_phone_(trans_model.NumPhones() + 1) { 

  if (opts_.add_partial_unk_label_left) {
    MakeFilterFst();
  }

  if (opts_.convert_to_unconstrained) {
    KALDI_WARN << "--convert-to-unconstrained=true; "
               << "--left-tolerance and --right-tolerance will be ignored.";
  } else {
    MakeToleranceEnforcerFst();
  }
}

void SupervisionLatticeSplitter::LoadLattice(const Lattice &lat) {
  lat_ = lat;

  PrepareLattice();

  int32 num_states = lat_.NumStates();

  KALDI_ASSERT(num_states > 0);  // TODO: Might have to be skipped instead.
  int32 start_state = lat_.Start();

  // Lattice should be top-sorted and connected, so start-state must be 0.
  KALDI_ASSERT(start_state == 0 && "Expecting start-state to be 0");

  KALDI_ASSERT(num_states == lat_scores_.state_times.size());
  KALDI_ASSERT(lat_scores_.state_times[start_state] == 0);
}

bool SupervisionLatticeSplitter::GetFrameRangeSupervision(
    int32 begin_frame, int32 num_frames,
    Supervision *supervision,
    Lattice *out_lat, Lattice *raw_range_lat) const {
  int32 end_frame = begin_frame + num_frames;
  // Note: end_frame is not included in the range of frames that the
  // output supervision object covers; it's one past the end.
  KALDI_ASSERT(num_frames > 0 && begin_frame >= 0 &&
               begin_frame + num_frames <= lat_scores_.state_times.back());

  Lattice lat_out;
  CreateRangeLattice(begin_frame, end_frame, &lat_out);

  if (raw_range_lat) {
    *raw_range_lat = lat_out;
  }

  PostProcessLattice(&lat_out);

  if (out_lat) {
    *out_lat = lat_out;
  }

  // Apply lm-scale on the lattice and remove the acoustic costs
  ScaleLattice(fst::LatticeScale(sup_opts_.lm_scale, 0.0), &lat_out);

  supervision->frames_per_sequence = num_frames;
  return GetSupervision(lat_out, supervision);
}

bool SupervisionLatticeSplitter::GetFrameRangeProtoSupervision(
    const ContextDependencyInterface &ctx_dep, 
    const TransitionModel &trans_model,
    int32 begin_frame, int32 num_frames,
    ProtoSupervision *proto_supervision) const {
  
  int32 end_frame = begin_frame + num_frames;
  // Note: end_frame is not included in the range of frames that the
  // output supervision object covers; it's one past the end.
  KALDI_ASSERT(num_frames > 0 && begin_frame >= 0 &&
               begin_frame + num_frames <= lat_scores_.state_times.back());

  Lattice lat_out;
  CreateRangeLattice(begin_frame, end_frame, &lat_out);
  
  PostProcessLattice(&lat_out);
  
  if (opts_.debug && GetVerboseLevel() > 2) {
    WriteLattice(std::cerr, false, lat_out);
  }

  CompactLattice clat_part;
  ConvertLattice(lat_out, &clat_part);

    
  return PhoneLatticeToProtoSupervision(sup_opts_, clat_part, 
                                        proto_supervision);
}

void SupervisionLatticeSplitter::LatticeInfo::Check() const {
  // Check if all the vectors are of size num_states
  KALDI_ASSERT(state_times.size() == alpha.size() &&
               state_times.size() == beta.size());

  // Check that the states are ordered in increasing order of state_times.
  // This must be true since the states are in breadth-first search order.
  KALDI_ASSERT(IsSorted(state_times));

  KALDI_ASSERT(state_times.back() == num_frames);
}

void SupervisionLatticeSplitter::PrepareLattice() {
  // Scale the lattice to appropriate acoustic scale.  
  KALDI_ASSERT(opts_.acoustic_scale != 0.0);
  if (opts_.acoustic_scale != 1.0)
    fst::ScaleLattice(fst::AcousticLatticeScale(
        opts_.acoustic_scale), &lat_);

  KALDI_ASSERT(fst::TopSort(&lat_));
  LatticeStateTimes(lat_, &(lat_scores_.state_times));
  int32 num_states = lat_.NumStates();
  std::vector<std::pair<int32,int32> > state_time_indexes(num_states);
  for (int32 s = 0; s < num_states; s++) {
    state_time_indexes[s] = std::make_pair(lat_scores_.state_times[s], s);
  }

  // Order the states based on the state times. This is stronger than just
  // topological sort. This is required by the lattice splitting code.
  std::sort(state_time_indexes.begin(), state_time_indexes.end());

  std::vector<int32> state_order(num_states);
  for (int32 s = 0; s < num_states; s++) {
    state_order[state_time_indexes[s].second] = s;
  }

  fst::StateSort(&lat_, state_order);
  ComputeLatticeScores();
}

void SupervisionLatticeSplitter::CreateRangeLattice(
    int32 begin_frame, int32 end_frame, 
    Lattice *out_lat) const {
  typedef Lattice::StateId StateId;
  typedef LatticeArc::Label Label;

  const std::vector<int32> &state_times = lat_scores_.state_times;

  // Some checks to ensure the lattice and scores are prepared properly
  KALDI_ASSERT(state_times.size() == lat_.NumStates());
  if (!lat_.Properties(fst::kTopSorted, true))
    KALDI_ERR << "Input lattice must be topologically sorted.";

  std::vector<int32>::const_iterator begin_iter =
      std::lower_bound(state_times.begin(), state_times.end(), begin_frame),
      end_iter = std::lower_bound(begin_iter,
                                  state_times.end(), end_frame);

  // begin_iter should point to the first state with time == begin_frame
  KALDI_ASSERT(*begin_iter == begin_frame &&
               (begin_iter == state_times.begin() ||
                begin_iter[-1] < begin_frame));

  // even if end_frame == supervision_.num_frames, there should be a state with
  // that frame index.
  KALDI_ASSERT(end_iter[-1] < end_frame &&
               (end_iter < state_times.end() || *end_iter == end_frame));
  
  StateId begin_state = begin_iter - state_times.begin(),
          end_state = end_iter - state_times.begin();

  KALDI_ASSERT(end_state > begin_state);
  out_lat->DeleteStates();
  out_lat->ReserveStates(end_state - begin_state + 2);

  // Add special start state
  StateId start_state = out_lat->AddState();
  out_lat->SetStart(start_state);

  KALDI_ASSERT(out_lat->Start() == 0);

  for (StateId i = begin_state; i < end_state; i++)
    out_lat->AddState();

  // Add the special final-state.
  StateId final_state = out_lat->AddState();
  out_lat->SetFinal(final_state, LatticeWeight::One());
  
  StateId prefinal_state = final_state + 1;
  bool need_prefinal_state = false;
  
  for (StateId state = begin_state; state < end_state; state++) {
    StateId output_state = state - begin_state + 1;
    if (state_times[state] == begin_frame) {
      // we'd like to make this an initial state, but OpenFst doesn't allow
      // multiple initial states.  Instead we add an epsilon transition to it
      // from our actual initial state.  The weight on this
      // transition is the forward probability of the said 'initial state'
      LatticeWeight weight = LatticeWeight::One();
      weight.SetValue1((opts_.normalize ? lat_scores_.beta[0] : 0.0) 
                        - lat_scores_.alpha[state]);
      // Add negative of the forward log-probability to the graph cost score,
      // since the acoustic scores would be changed later.
      // Assuming that the lattice is scaled with appropriate acoustic
      // scale.
      // We additionally normalize using the total lattice score. Since the
      // same score is added as normalizer to all the paths in the lattice,
      // the relative probabilities of the paths in the lattice is not affected.
      // Note: Doing a forward-backward on this split must result in a total
      // score of 0 because of the normalization.

      out_lat->AddArc(start_state,
                      LatticeArc(0, 0, weight, output_state));
    } else {
      KALDI_ASSERT(lat_scores_.state_times[state] < end_frame);
    }
    for (fst::ArcIterator<Lattice> aiter(lat_, state);
          !aiter.Done(); aiter.Next()) {
      const LatticeArc &arc = aiter.Value();
      StateId nextstate = arc.nextstate;
      if (nextstate >= end_state) {
        // A transition to any state outside the range becomes a transition to
        // our special final-state.
        // The weight is just the negative of the backward log-probability +
        // the arc cost. We again normalize with the total lattice score.
        LatticeWeight weight;
        //KALDI_ASSERT(lat_scores_.beta[state] < 0);
        weight.SetValue1(arc.weight.Value1() - lat_scores_.beta[nextstate]);
        weight.SetValue2(arc.weight.Value2());
        // Add negative of the backward log-probability to the LM score, since
        // the acoustic scores would be changed later.
        // Note: We don't normalize here because that is already done with the
        // initial cost.

        if (!opts_.add_partial_unk_label_left) {
          out_lat->AddArc(output_state,
              LatticeArc(arc.ilabel, arc.olabel, weight, final_state));
        } else {
          fst::ArcIterator<Lattice> next_aiter(lat_, nextstate);
          if (!next_aiter.Done() && next_aiter.Value().olabel == 0) {
            // This is a split in the middle of a phone.
            // So add an arc to the "prefinal state" from which there 
            // is an arc to the "final state" with special 
            // "incomplete phone" symbol on the output-label.
            
            if (!need_prefinal_state) {
              prefinal_state = out_lat->AddState();
              need_prefinal_state = true;
            }

            out_lat->AddArc(output_state,
                LatticeArc(arc.ilabel, arc.olabel, weight, prefinal_state));
          } else {
            out_lat->AddArc(output_state,
                LatticeArc(arc.ilabel, arc.olabel, weight, final_state));
          }
        } 
      } else {
        StateId output_nextstate = nextstate - begin_state + 1;

        Label olabel = arc.olabel;

        if (state_times[state] == begin_frame &&
            (opts_.add_partial_phone_label_right ||
             opts_.add_partial_unk_label_right)) {
          int32 tid = arc.ilabel;
          int32 phone = trans_model_.TransitionIdToPhone(tid);

          if (opts_.add_partial_unk_label_right) {
            KALDI_ASSERT(opts_.unk_phone > 0);
            phone = opts_.unk_phone;
          }

          if (olabel == 0) {
            // This is a split in the middle of a phone.
            // So add a phone label as the output label.
            olabel = phone;
          }
        }
        out_lat->AddArc(output_state,
            LatticeArc(arc.ilabel, olabel, arc.weight, output_nextstate));
      }
    }
  }
  
  if (need_prefinal_state) {
    // Add an "incomplete phone" label as the output symbol in the 
    // last arc
    out_lat->AddArc(prefinal_state, 
        LatticeArc(0, incomplete_phone_, LatticeWeight::One(),
                   final_state));
  }
  
  KALDI_ASSERT(out_lat->Start() == 0);

  if (opts_.debug) {
    Posterior post;

    Lattice &temp_lat(*out_lat);
    //fst::RmEpsilon(&temp_lat);
    fst::TopSort(&temp_lat);

    double like = LatticeForwardBackward(temp_lat, &post);

    KALDI_ASSERT(kaldi::ApproxEqual(
          like + (opts_.normalize ? lat_scores_.beta[0] : 0.0),
          lat_scores_.beta[0]));

    const Posterior &full_post = lat_scores_.post;

    for (int32 t = begin_frame; t < end_frame; t++) {
      KALDI_ASSERT(full_post[t].size() == post[t - begin_frame].size());

      for (int32 j = 0; j < full_post[t].size(); j++) {
        KALDI_ASSERT(post[t - begin_frame][j].first == full_post[t][j].first);
        if (post[t-begin_frame][j].second < 0.1)
          continue;
        if (!kaldi::ApproxEqual(post[t - begin_frame][j].second, 
                                        full_post[t][j].second)) {
          WritePosterior(std::cerr, false, full_post);
          WritePosterior(std::cerr, false, post);

          std::vector<double> alphas;
          std::vector<double> betas;
          ComputeLatticeAlphasAndBetas(temp_lat, false, &alphas, &betas);

          fst::StdVectorFst full_fst;
          Lattice full_lat(lat_);
          fst::ScaleLattice(fst::AcousticLatticeScale(0), &full_lat);
          ConvertLattice(full_lat, &full_fst);
          WriteFstKaldi(std::cerr, false, full_fst);
          
          fst::StdVectorFst split_fst;
          fst::ScaleLattice(fst::AcousticLatticeScale(0), out_lat);
          ConvertLattice(*out_lat, &split_fst);
          WriteFstKaldi(std::cerr, false, split_fst);

          KALDI_ASSERT(false);
        }
      }
    }
  }
}

void SupervisionLatticeSplitter::PostProcessLattice(Lattice *out_lat) const {
  if (opts_.add_partial_unk_label_left) {
    if (opts_.debug && GetVerboseLevel() > 2) {
      WriteLattice(std::cerr, false, *out_lat);
    }

    fst::TableComposeOptions compose_opts;
    compose_opts.table_match_type = fst::MATCH_OUTPUT;

    Lattice filter_lat;
    FstToLattice(filter_fst_, &filter_lat);

    Lattice temp_lat;
    TableCompose(*out_lat, filter_lat, &temp_lat);

    std::swap(temp_lat, *out_lat);
    
    if (opts_.debug && GetVerboseLevel() > 2) {
      WriteLattice(std::cerr, false, *out_lat);
    }
  }
  
  fst::RmEpsilon(out_lat);

  if (opts_.acoustic_scale != 1.0) {
    fst::ScaleLattice(fst::AcousticLatticeScale(
          1.0 / opts_.acoustic_scale), out_lat);
  }
}

bool SupervisionLatticeSplitter::GetSupervision(
    const Lattice &lat, Supervision *supervision) const {
  fst::StdVectorFst transition_id_fst;
  ConvertLattice(lat, &transition_id_fst);
  Project(&transition_id_fst, fst::PROJECT_INPUT);  // Keep only the transition-ids.
  if (transition_id_fst.Properties(fst::kIEpsilons, true) != 0) {
    // remove epsilons, if there are any.
    fst::RmEpsilon(&transition_id_fst);
  }

  KALDI_ASSERT(transition_id_fst.NumStates() > 0);

  if (opts_.convert_to_unconstrained) {
    supervision->label_dim = trans_model_.NumTransitionIds();
    std::swap(transition_id_fst, supervision->fst);
    return ConvertSupervisionToUnconstrained(trans_model_, supervision);
  } else {
    supervision->label_dim = trans_model_.NumPdfs();
  }

  fst::TableComposeOptions compose_opts;
  compose_opts.table_match_type = fst::MATCH_INPUT;

  TableCompose(transition_id_fst, tolerance_fst_, &(supervision->fst),
               compose_opts);

  fst::Connect(&(supervision->fst));

  // at this point supervision->fst will have pdf-ids plus one as the olabels,
  // but still transition-ids as the ilabels.  Copy olabels to ilabels.
  fst::Project(&(supervision->fst), fst::PROJECT_OUTPUT);

  fst::RmEpsilon(&(supervision->fst));
  fst::DeterminizeInLog(&(supervision->fst));

  if (opts_.debug) {
    std::cerr << "tolerance added fst";
    fst::WriteFstKaldi(std::cerr, false, supervision->fst);
  }

  KALDI_ASSERT(supervision->fst.Properties(fst::kIEpsilons, true) == 0);
  if (supervision->fst.NumStates() == 0) {
    KALDI_WARN << "Supervision FST is empty (too many phones for too few "
               << "frames?)";
    // possibly there were too many phones for too few frames.
    return false;
  }

  supervision->weight = 1.0;
  supervision->num_sequences = 1;
  supervision->label_dim = trans_model_.NumPdfs();
  SortBreadthFirstSearch(&(supervision->fst));

  return true;
}

void SupervisionLatticeSplitter::ComputeLatticeScores() {
  lat_scores_.Reset();
  lat_scores_.num_frames = LatticeStateTimes(lat_, &(lat_scores_.state_times));

  if (opts_.debug)
    LatticeForwardBackward(lat_, &(lat_scores_.post));

  ComputeLatticeAlphasAndBetas(lat_, false,
                               &(lat_scores_.alpha), &(lat_scores_.beta));
  lat_scores_.Check();
  // This check will fail if the lattice is not breadth-first search sorted
}

class ToleranceEnforcerFstCreator {
 public:
  ToleranceEnforcerFstCreator(
      const SupervisionOptions &opts, const TransitionModel &trans_model,
      fst::StdVectorFst *fst);
  
  void MakeFst();

 private:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  enum StateType {
    kInit,
    kDeletion,
    kAccept,
    kInsertion
  };

  inline int32 GetStateId(int32 offset, int32 forward_id, int32 type) {
    return ((offset + zero_offset_index_) * (num_forward_transitions_ * 3 + 1)
            + (type == kInit ? 0 : 1 + (type - 1) * num_forward_transitions_
                                   + forward_id) + 1);
  }

  void AddArcsForOffset(int32 offset);
  void AddArcsForForwardTransition(int32 offset, int32 forward_id, int32 trans_id);
  void AddArcsBetweenOffsets(int32 offset, int32 forward_id, int32 trans_id);

  const SupervisionOptions &opts_;
  const TransitionModel &trans_model_;

  int32 num_forward_transitions_;  // number of forward transitions in the 
                                   // transition model
  int32 num_offsets_;  // number of offsets (tolerances)

  // The index corresponding to the zero offset.
  // offset_index = offset + zero_offset_index_
  int32 zero_offset_index_; 

  fst::StdVectorFst *fst_;
};
  
ToleranceEnforcerFstCreator::ToleranceEnforcerFstCreator(
    const SupervisionOptions &opts, const TransitionModel &trans_model,
    fst::StdVectorFst *fst):
    opts_(opts), trans_model_(trans_model), fst_(fst) {

  num_forward_transitions_ = 0;
  for (int32 trans_id = 1; trans_id <= trans_model_.NumTransitionIds(); 
      trans_id++) {
    if (!trans_model_.IsSelfLoop(trans_id)) {
      num_forward_transitions_++;
    }
  }
  num_offsets_ = opts_.left_tolerance + opts_.right_tolerance + 1;
  zero_offset_index_ = opts_.left_tolerance;

  fst_->DeleteStates();
}

void ToleranceEnforcerFstCreator::AddArcsForForwardTransition(
    int32 offset, int32 forward_id, int32 trans_id) {
  StateId init_state = GetStateId(offset, forward_id, kInit);

  if (offset == 0 && forward_id == 0)
    fst_->SetFinal(init_state, fst::TropicalWeight::One());

  // We expect this is to be a forward transition
  KALDI_ASSERT(!trans_model_.IsSelfLoop(trans_id));
  int32 forward_pdf = trans_model_.TransitionIdToPdf(trans_id);
  int32 tstate = trans_model_.TransitionIdToTransitionState(trans_id);
  int32 self_loop_tid = trans_model_.SelfLoopOf(tstate);
  int32 self_loop_pdf = trans_model_.TransitionIdToPdf(self_loop_tid);

  // self-loop accepting forward-tid
  fst_->AddArc(init_state,
               fst::StdArc(trans_id, forward_pdf + 1,
                           fst::TropicalWeight::One(),
                           init_state));
                          
  for (int32 i = 1; i <= 3; i++) {
    StateId next_state = GetStateId(offset, forward_id, i);

    if (i == kDeletion || i == kInsertion) {
      // epsilon-arc to initial state
      fst_->AddArc(next_state,
                   fst::StdArc(0, 0,
                               fst::TropicalWeight::One(),
                               init_state));
    }

    if (i == kAccept) {
      // accept a forward transition from initial state
      fst_->AddArc(init_state, 
                   fst::StdArc(trans_id, forward_pdf + 1,
                               fst::TropicalWeight::One(),
                               next_state));

      // self-loop accepting self-loop tid
      fst_->AddArc(next_state,
                   fst::StdArc(self_loop_tid, self_loop_pdf + 1,
                               fst::TropicalWeight::One(),
                               next_state));
                                
      // self-loop transition back to initial state
      fst_->AddArc(next_state,
                   fst::StdArc(self_loop_tid, self_loop_pdf + 1,
                               fst::TropicalWeight::One(),
                               init_state));
    }
  }
}

void ToleranceEnforcerFstCreator::AddArcsBetweenOffsets(
    int32 offset, int32 forward_id, int32 trans_id) {
  // We expect this is to be a forward transition
  KALDI_ASSERT(!trans_model_.IsSelfLoop(trans_id));
  int32 tstate = trans_model_.TransitionIdToTransitionState(trans_id);
  int32 self_loop_tid = trans_model_.SelfLoopOf(tstate);
  int32 self_loop_pdf = trans_model_.TransitionIdToPdf(self_loop_tid);

  if (offset > -opts_.left_tolerance) {
    StateId accept_state = GetStateId(offset, forward_id, kAccept);
    StateId delete_state = GetStateId(offset, forward_id, kDeletion);
    StateId next_state = GetStateId(offset - 1, forward_id, kDeletion);

    fst_->AddArc(accept_state, 
                 fst::StdArc(self_loop_tid, 0,
                             fst::TropicalWeight::One(),
                             next_state));
    fst_->AddArc(delete_state, 
                 fst::StdArc(self_loop_tid, 0,
                             fst::TropicalWeight::One(),
                             next_state));
  } 

  if (offset < opts_.right_tolerance) {
    StateId accept_state = GetStateId(offset, forward_id, kAccept);
    StateId insert_state = GetStateId(offset, forward_id, kInsertion);
    StateId next_state = GetStateId(offset + 1, forward_id, kInsertion);

    fst_->AddArc(accept_state, 
                 fst::StdArc(0, self_loop_pdf + 1,
                             fst::TropicalWeight::One(),
                             next_state));
    
    fst_->AddArc(insert_state, 
                 fst::StdArc(0, self_loop_pdf + 1,
                             fst::TropicalWeight::One(),
                             next_state));
  }
  
  if (offset == 0) {
    if (forward_id == 0) {
      // Add arc from start state to the offset 0 initial state.
      // This is the normal case when there is no partial phone in the lattice.
      StateId init_state = GetStateId(offset, forward_id, kInit);
      fst_->AddArc(0, fst::StdArc(0, 0,
                                  fst::TropicalWeight::One(),
                                  init_state));
    }
    
    // Add self-loop on start state accepting the self-loop transition 
    // of a partial phone.
    fst_->AddArc(0, fst::StdArc(self_loop_tid, self_loop_pdf + 1,
                                fst::TropicalWeight::One(),
                                0));

    if (offset > -opts_.left_tolerance) {
      // Add arc from start state deleting a self-loop transition
      StateId next_state = GetStateId(offset - 1, forward_id, kDeletion);
      fst_->AddArc(0, fst::StdArc(self_loop_tid, 0,
                                  fst::TropicalWeight::One(),
                                  next_state));
    }
  }
}

void ToleranceEnforcerFstCreator::AddArcsForOffset(int32 offset) {
  int32 forward_id = 0;
  for (int32 trans_id = 1; trans_id <= trans_model_.NumTransitionIds(); 
       trans_id++) {
    if (!trans_model_.IsSelfLoop(trans_id)) {
      AddArcsForForwardTransition(offset, forward_id, trans_id);
      AddArcsBetweenOffsets(offset, forward_id, trans_id);
      forward_id++;
    }
  }
}

void ToleranceEnforcerFstCreator::MakeFst() {
  int32 num_states = num_offsets_ * (3 * num_forward_transitions_ + 1) + 1;
  fst_->ReserveStates(num_states);
 
  for (int32 s = 0; s < num_states; s++)
    fst_->AddState();

  fst_->SetStart(0);

  for (int32 o = -opts_.left_tolerance; o <= opts_.right_tolerance; o++) {
    AddArcsForOffset(o);
  }
  
  if (GetVerboseLevel() > 3) { WriteFstKaldi(std::cerr, false, *fst_); }

  fst::Connect(fst_);
  fst::ArcSort(fst_, fst::ILabelCompare<fst::StdArc>());
}

void SupervisionLatticeSplitter::MakeToleranceEnforcerFst() {
  GetToleranceEnforcerFst(sup_opts_, trans_model_, &tolerance_fst_);
}

void GetToleranceEnforcerFst(const SupervisionOptions &sup_opts,
                              const TransitionModel &trans_model,
                              fst::StdVectorFst *tolerance_fst) {
  ToleranceEnforcerFstCreator creator(sup_opts, trans_model, tolerance_fst);
  creator.MakeFst();
}

void SupervisionLatticeSplitter::MakeFilterFst() {
  filter_fst_.DeleteStates();
  filter_fst_.AddState();
  filter_fst_.AddState();
  filter_fst_.AddState();

  filter_fst_.SetStart(0);

  const std::vector<int32> &phones = trans_model_.GetPhones();
  for (std::vector<int32>::const_iterator it = phones.begin();
       it != phones.end(); ++it) {
    filter_fst_.AddArc(0, fst::StdArc(*it, *it,
                                      fst::TropicalWeight::One(), 0));
    filter_fst_.AddArc(0, fst::StdArc(*it, opts_.unk_phone,
                                      fst::TropicalWeight::One(), 1));
  }
  filter_fst_.AddArc(1, fst::StdArc(incomplete_phone_, 0, 
                                    fst::TropicalWeight::One(), 2));
  
  filter_fst_.SetFinal(0, fst::TropicalWeight::One());
  filter_fst_.SetFinal(2, fst::TropicalWeight::One());

  fst::ArcSort(&filter_fst_, fst::ILabelCompare<fst::StdArc>());
}

/*
bool PhoneLatticeToSupervision(const fst::StdVectorFst &tolerance_fst,
                               const TransitionModel &trans_model,
                               const Lattice &lat,
                               chain::Supervision *supervision,
                               bool debug) {
  fst::StdVectorFst transition_id_fst;
  ConvertLattice(lat, &transition_id_fst);
  Project(&transition_id_fst, fst::PROJECT_INPUT);  // Keep only the transition-ids.
  if (transition_id_fst.Properties(fst::kIEpsilons, true) != 0) {
    // remove epsilons, if there are any.
    fst::RmEpsilon(&transition_id_fst);
  }
  KALDI_ASSERT(transition_id_fst.NumStates() > 0);

  fst::TableComposeOptions compose_opts;
  compose_opts.table_match_type = fst::MATCH_INPUT;

  TableCompose(transition_id_fst, tolerance_fst, &(supervision->fst),
               compose_opts);
  fst::Connect(&(supervision->fst));

  if (debug) {
    fst::Project(&(supervision->fst), fst::PROJECT_OUTPUT);
    fst::RmEpsilon(&(supervision->fst));
    
    return true;
  }

  // at this point supervision->fst will have pdf-ids plus one as the olabels,
  // but still transition-ids as the ilabels.  Copy olabels to ilabels.
  fst::Project(&(supervision->fst), fst::PROJECT_OUTPUT);
  
  fst::RmEpsilon(&(supervision->fst));
  fst::DeterminizeInLog(&(supervision->fst));

  KALDI_ASSERT(supervision->fst.Properties(fst::kIEpsilons, true) == 0);
  if (supervision->fst.NumStates() == 0) {
    KALDI_WARN << "Supervision FST is empty (too many phones for too few "
               << "frames?)";
    // possibly there were too many phones for too few frames.
    return false;
  }

  supervision->weight = 1.0;
  supervision->num_sequences = 1;
  supervision->frames_per_sequence = 0;
  supervision->label_dim = trans_model.NumPdfs();
  SortBreadthFirstSearch(&(supervision->fst));
  return true;
}
*/

}  // end namespace chain
}  // end namespace kaldi
