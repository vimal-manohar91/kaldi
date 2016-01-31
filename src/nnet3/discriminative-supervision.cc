// nnet3/discriminative-supervision.cc

// Copyright 2012-2015  Johns Hopkins University (author: Daniel Povey)
//           2014-2015  Vimal Manohar

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

#include "nnet3/discriminative-supervision.h"
#include "lat/lattice-functions.h"

namespace kaldi {
namespace discriminative {

void DiscriminativeSupervisionOptions::Check() const {
  KALDI_ASSERT(frame_subsampling_factor > 0);
}

DiscriminativeSupervision::DiscriminativeSupervision(const DiscriminativeSupervision &other):
    weight(other.weight), num_sequences(other.num_sequences),
    frames_per_sequence(other.frames_per_sequence), 
    num_ali(other.num_ali), oracle_ali(other.oracle_ali),
    weights(other.weights),
    num_lat_present(other.num_lat_present),
    num_lat(other.num_lat),
    den_lat(other.den_lat) { }

void DiscriminativeSupervision::Swap(DiscriminativeSupervision *other) {
  std::swap(weight, other->weight);
  std::swap(num_sequences, other->num_sequences);
  std::swap(frames_per_sequence, other->frames_per_sequence);
  std::swap(num_ali, other->num_ali);
  std::swap(oracle_ali, other->oracle_ali);
  std::swap(weights, other->weights);
  std::swap(num_lat_present, other->num_lat_present);
  std::swap(num_lat, other->num_lat);
  std::swap(den_lat, other->den_lat);
}

bool DiscriminativeSupervision::operator == (const DiscriminativeSupervision &other) const {
  return ( weight == other.weight && num_sequences == other.num_sequences &&
      frames_per_sequence == other.frames_per_sequence &&
      num_ali == other.num_ali &&
      oracle_ali == other.oracle_ali &&
      weights == other.weights &&
      num_lat_present == other.num_lat_present &&
      fst::Equal(num_lat, other.num_lat) && 
      fst::Equal(den_lat, other.den_lat) );
}

void DiscriminativeSupervision::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<DiscriminativeSupervision>");
  WriteToken(os, binary, "<Weight>");
  WriteBasicType(os, binary, weight);
  WriteToken(os, binary, "<NumSequences>");
  WriteBasicType(os, binary, num_sequences);
  WriteToken(os, binary, "<FramesPerSeq>");
  WriteBasicType(os, binary, frames_per_sequence);
  KALDI_ASSERT(frames_per_sequence > 0 &&
               num_sequences > 0);
  
  WriteToken(os, binary, "<NumAli>");
  WriteIntegerVector(os, binary, num_ali);

  CompactLattice clat;
  if (num_lat_present) {
    WriteToken(os, binary, "<NumLat>");
    ConvertLattice(num_lat, &clat);
    if (!WriteCompactLattice(os, binary, clat)) {
      KALDI_ERR << "Error writing numerator lattice to stream";
    }
  } 

  WriteToken(os, binary, "<OracleAli>");
  WriteIntegerVector(os, binary, oracle_ali);

  WriteToken(os, binary, "<FrameWeights>");
  Vector<BaseFloat> frame_weights(weights.size());
  for (size_t i = 0; i < weights.size(); i++) {
    frame_weights(i) = weights[i];
  }
  frame_weights.Write(os, binary);

  WriteToken(os, binary, "<DenLat>");
  ConvertLattice(den_lat, &clat);
  if (!WriteCompactLattice(os, binary, clat)) {
    // We can't return error status from this function so we
    // throw an exception. 
    KALDI_ERR << "Error writing denominator lattice to stream";
  }

  WriteToken(os, binary, "</DiscriminativeSupervision>");
}

void DiscriminativeSupervision::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<DiscriminativeSupervision>");
  ExpectToken(is, binary, "<Weight>");
  ReadBasicType(is, binary, &weight);
  ExpectToken(is, binary, "<NumSequences>");
  ReadBasicType(is, binary, &num_sequences);
  ExpectToken(is, binary, "<FramesPerSeq>");
  ReadBasicType(is, binary, &frames_per_sequence);
  KALDI_ASSERT(frames_per_sequence > 0 && 
               num_sequences > 0);
  
  ExpectToken(is, binary, "<NumAli>");
  ReadIntegerVector(is, binary, &num_ali);

  std::string token;
  ReadToken(is, binary, &token);

  if (token == "<NumLat>") {
    num_lat_present = true;
    CompactLattice *clat = NULL;
    if (!ReadCompactLattice(is, binary, &clat) || clat == NULL) {
      // We can't return error status from this function so we
      // throw an exception. 
      KALDI_ERR << "Error reading CompactLattice from stream";
    }
    ConvertLattice(*clat, &num_lat);
    delete clat;
    TopSort(&num_lat);
    ReadToken(is, binary, &token);
  } 
 
  if (token != "<OracleAli>") {
    KALDI_ERR << "Expecting token <OracleAli>; got token " << token;
  }
  
  ReadIntegerVector(is, binary, &oracle_ali);

  ExpectToken(is, binary, "<FrameWeights>");
  Vector<BaseFloat> frame_weights;
  frame_weights.Read(is, binary);
  weights.clear();
  std::copy(frame_weights.Data(), frame_weights.Data() + frame_weights.Dim(), std::back_inserter(weights));
  
  ExpectToken(is, binary, "<DenLat>");
  {
    CompactLattice *clat = NULL;
    if (!ReadCompactLattice(is, binary, &clat) || clat == NULL) {
      // We can't return error status from this function so we
      // throw an exception. 
      KALDI_ERR << "Error reading CompactLattice from stream";
    }
    ConvertLattice(*clat, &den_lat);
    delete clat;
    TopSort(&den_lat);
  }

  ExpectToken(is, binary, "</DiscriminativeSupervision>");
}

bool LatticeToDiscriminativeSupervision(const std::vector<int32> &num_ali,
                                        const CompactLattice &num_lat, 
                                        const CompactLattice &den_lat,
                                        BaseFloat weight,
                                        DiscriminativeSupervision *supervision,
                                        const Vector<BaseFloat> *weights,
                                        const std::vector<int32> *oracle_alignment) {
  supervision->weight = weight;
  supervision->num_sequences = 1;
  supervision->frames_per_sequence = num_ali.size();
  supervision->num_ali = num_ali;
  supervision->num_lat_present = true;
  ConvertLattice(num_lat, &supervision->num_lat);
  TopSort(&(supervision->num_lat));
  ConvertLattice(den_lat, &supervision->den_lat);
  TopSort(&(supervision->den_lat));
  if (weights) {
    supervision->weights.clear();
    std::copy(weights->Data(), weights->Data() + weights->Dim(), 
              std::back_inserter(supervision->weights));
  }
  if (oracle_alignment)
    supervision->oracle_ali = *oracle_alignment;

  supervision->Check();

  return true;
}

bool LatticeToDiscriminativeSupervision(const std::vector<int32> &num_ali,
                                        const CompactLattice &den_lat, 
                                        BaseFloat weight,
                                        DiscriminativeSupervision *supervision,
                                        const Vector<BaseFloat> *weights,
                                        const std::vector<int32> *oracle_alignment) {
  supervision->weight = weight;
  supervision->num_sequences = 1;
  supervision->frames_per_sequence = num_ali.size();
  supervision->num_ali = num_ali;
  supervision->num_lat_present = false;
  ConvertLattice(den_lat, &supervision->den_lat);
  TopSort(&(supervision->den_lat));
  if (weights) {
    supervision->weights.clear();
    std::copy(weights->Data(), weights->Data() + weights->Dim(), 
        std::back_inserter(supervision->weights));
  }
  if (oracle_alignment)
    supervision->oracle_ali = *oracle_alignment;

  supervision->Check();

  return true;
}

void DiscriminativeSupervision::Check() const {
  int32 num_frames = frames_per_sequence * num_sequences;

  KALDI_ASSERT(static_cast<int32> (num_ali.size()) == num_frames);
  KALDI_ASSERT(oracle_ali.size() == 0 || 
               static_cast<int32> (oracle_ali.size()) == num_frames);
  KALDI_ASSERT(weights.size() == 0 || 
               static_cast<int32> (weights.size()) == num_frames);
  
  {
    std::vector<int32> state_times;
    int32 max_time = LatticeStateTimes(den_lat, &state_times);
    KALDI_ASSERT(max_time == num_frames);
  }

  if (num_lat_present) {
    std::vector<int32> state_times;
    int32 max_time = LatticeStateTimes(num_lat, &state_times);
    KALDI_ASSERT(max_time == num_frames);
  }
}

DiscriminativeSupervisionSplitter::DiscriminativeSupervisionSplitter(
    const SplitDiscriminativeSupervisionOptions &config,
    const TransitionModel &tmodel,
    const DiscriminativeSupervision &supervision):
    config_(config), tmodel_(tmodel), supervision_(supervision), 
    num_lat_present_(supervision.num_lat_present) {
  if (supervision_.num_sequences != 1) {
    KALDI_WARN << "Splitting already-reattached sequence (only expected in "
               << "testing code)";
  }

  KALDI_ASSERT(supervision_.num_sequences == 1); // For now, don't allow splitting already merged examples

  // Prepare lattice : 
  // 1) Order states in breadth-first search order
  // 2) Compute states times, which must be a strictly non-decreasing vector
  // 3) Compute lattice alpha and beta scores

  den_lat_ = supervision_.den_lat;
  PrepareLattice(&den_lat_, &den_lat_scores_);
  
  if (supervision_.num_lat_present) {
    num_lat_ = supervision_.num_lat;
    PrepareLattice(&num_lat_, &num_lat_scores_);
  }
  
  int32 num_states = den_lat_.NumStates(),
        num_frames = supervision_.frames_per_sequence * supervision_.num_sequences;
  KALDI_ASSERT(num_states > 0);
  int32 start_state = den_lat_.Start();
  // Lattice should be top-sorted and connected, so start-state must be 0.
  KALDI_ASSERT(start_state == 0 && "Expecting start-state to be 0");
  
  // The following asserts checks that the number of frames in the lattice 
  // matches the num_frames stored in the supervision object; 
  // it also relies on the breadth-first search sorting and connectedness
  // of the FST.
  if (num_lat_present_) {
    KALDI_ASSERT(num_states == num_lat_scores_.state_times.size());
    KALDI_ASSERT(num_lat_scores_.state_times[start_state] == 0);
    KALDI_ASSERT(num_lat_scores_.state_times.back() == num_frames);
  }

  KALDI_ASSERT(num_states == den_lat_scores_.state_times.size());
  KALDI_ASSERT(den_lat_scores_.state_times[start_state] == 0);
  KALDI_ASSERT(den_lat_scores_.state_times.back() == num_frames);
}

// Make sure that for any given pdf-id and any given frame, the den-lat has
// only one transition-id mapping to that pdf-id, on the same frame.
// It helps us to more completely minimize the lattice.  Note: we
// can't do this if the criterion is MPFE, because in that case the
// objective function will be affected by the phone-identities being
// different even if the pdf-ids are the same.
void DiscriminativeSupervisionSplitter::CollapseTransitionIds(
    const std::vector<int32> &state_times, Lattice *lat) const {
  typedef Lattice::StateId StateId;
  typedef Lattice::Arc Arc;

  int32 num_frames = state_times.back();   // TODO: Check if this is always true
  StateId num_states = lat->NumStates();

  std::vector<std::map<int32, int32> > pdf_to_tid(num_frames);
  for (StateId s = 0; s < num_states; s++) {
    int32 t = state_times[s];
    for (fst::MutableArcIterator<Lattice> aiter(lat, s);
         !aiter.Done(); aiter.Next()) {
      KALDI_ASSERT(t >= 0 && t < num_frames);
      Arc arc = aiter.Value();
      KALDI_ASSERT(arc.ilabel != 0 && arc.ilabel == arc.olabel);
      int32 pdf = tmodel_.TransitionIdToPdf(arc.ilabel);
      if (pdf_to_tid[t].count(pdf) != 0) {
        arc.ilabel = arc.olabel = pdf_to_tid[t][pdf];
        aiter.SetValue(arc);
      } else {
        pdf_to_tid[t][pdf] = arc.ilabel;
      }
    }
  }    
}

void DiscriminativeSupervisionSplitter::LatticeInfo::Check() const {
  // Check if all the vectors are of size num_states
  KALDI_ASSERT(state_times.size() == alpha_p.size() &&
               state_times.size() == beta_p.size());

  // Check that the states are ordered in increasing order of state_times
  int32 t = 0;
  for (std::vector<int32>::const_iterator it = state_times.begin();
          it != state_times.end(); ++it) {
    if (it == state_times.begin()) {
      t = *it;
      continue;
    }
    int32 cur_t = *it; 
    KALDI_ASSERT(cur_t >= t);   
    t = cur_t;
  }
} 

void DiscriminativeSupervisionSplitter::GetFrameRange(int32 begin_frame, int32 num_frames, bool normalize, 
                                                      DiscriminativeSupervision *out_supervision) const {
  int32 end_frame = begin_frame + num_frames;
  // Note: end_frame is not included in the range of frames that the
  // output supervision object covers; it's one past the end.
  KALDI_ASSERT(num_frames > 0 && begin_frame >= 0 &&
               begin_frame + num_frames <=
               supervision_.num_sequences * supervision_.frames_per_sequence);

  CreateRangeLattice(den_lat_,
                     den_lat_scores_,
                     begin_frame, end_frame, normalize,
                     &(out_supervision->den_lat));

  if (num_lat_present_) {
    CreateRangeLattice(num_lat_, 
                       num_lat_scores_,
                       begin_frame, end_frame, normalize,
                       &(out_supervision->num_lat));
  }
  out_supervision->num_lat_present = num_lat_present_;

  out_supervision->num_ali.clear();
  std::copy(supervision_.num_ali.begin() + begin_frame,
            supervision_.num_ali.begin() + end_frame,
            std::back_inserter(out_supervision->num_ali));
  
  out_supervision->oracle_ali.clear();
  if (supervision_.oracle_ali.size() > 0) {
    std::copy(supervision_.oracle_ali.begin() + begin_frame,
        supervision_.oracle_ali.begin() + end_frame,
        std::back_inserter(out_supervision->oracle_ali));
  }

  out_supervision->weights.clear();
  if (supervision_.weights.size() > 0) {
    std::copy(supervision_.weights.begin() + begin_frame,
        supervision_.weights.begin() + end_frame,
        std::back_inserter(out_supervision->weights));
  }

  out_supervision->num_sequences = 1;
  out_supervision->weight = supervision_.weight;
  out_supervision->frames_per_sequence = num_frames;

  out_supervision->Check();
}

void DiscriminativeSupervisionSplitter::CreateRangeLattice(
    const Lattice &in_lat, const LatticeInfo &scores,
    int32 begin_frame, int32 end_frame, bool normalize,
    Lattice *out_lat) const {
  typedef Lattice::StateId StateId;

  const std::vector<int32> &state_times = scores.state_times;
  
  // Some checks to ensure the lattice and scores are prepared properly 
  KALDI_ASSERT(state_times.size() == in_lat.NumStates());
  if (!in_lat.Properties(fst::kTopSorted, true))
    KALDI_ERR << "Input lattice must be topologically sorted.";

  std::vector<int32>::const_iterator begin_iter =
      std::lower_bound(state_times.begin(), state_times.end(), begin_frame),
      end_iter = std::lower_bound(begin_iter, 
                                  state_times.end(), end_frame);

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
  
  for (StateId i = begin_state; i < end_state; i++)
    out_lat->AddState();
  
  // Add the special final-state.
  StateId final_state = out_lat->AddState();
  out_lat->SetFinal(final_state, LatticeWeight::One());

  for (StateId state = begin_state; state < end_state; state++) {
    StateId output_state = state - begin_state + 1;
    if (state_times[state] == begin_frame) {
      // we'd like to make this an initial state, but OpenFst doesn't allow
      // multiple initial states.  Instead we add an epsilon transition to it
      // from our actual initial state.  The weight on this 
      // transition is the forward probability of the said 'initial state'
      LatticeWeight weight = LatticeWeight::One();
      weight.SetValue1((normalize ? scores.beta_p[0] : 0.0) - scores.alpha_p[state]); 
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
      KALDI_ASSERT(scores.state_times[state] < end_frame);
    }
    for (fst::ArcIterator<Lattice> aiter(in_lat, state); 
          !aiter.Done(); aiter.Next()) {
      const LatticeArc &arc = aiter.Value();
      StateId nextstate = arc.nextstate;
      if (nextstate >= end_state) {
        // A transition to any state outside the range becomes a transition to
        // our special final-state. 
        // The weight is just the negative of the backward log-probability + 
        // the arc cost. We again normalize with the total lattice score.
        LatticeWeight weight;
        //KALDI_ASSERT(scores.beta_p[state] < 0);
        weight.SetValue1(arc.weight.Value1() - scores.beta_p[nextstate]); 
        weight.SetValue2(arc.weight.Value2());
        // Add negative of the backward log-probability to the LM score, since
        // the acoustic scores would be changed later.
        // Note: We don't normalize here because that is already done with the
        // initial cost.
      
        out_lat->AddArc(output_state,
            LatticeArc(arc.ilabel, arc.olabel, weight, final_state));
      } else {
        StateId output_nextstate = nextstate - begin_state + 1;
        out_lat->AddArc(output_state,
            LatticeArc(arc.ilabel, arc.olabel, arc.weight, output_nextstate));
      }
    }
  }

  // Get rid of the word labels and put the
  // transition-ids on both sides.
  fst::Project(out_lat, fst::PROJECT_INPUT);
  fst::RmEpsilon(out_lat);

  if (config_.collapse_transition_ids)
    CollapseTransitionIds(state_times, out_lat);

  if (config_.determinize) {
    if (!config_.minimize) {
      Lattice tmp_lat;
      fst::Determinize(*out_lat, &tmp_lat);
      std::swap(*out_lat, tmp_lat);
    } else {
      Lattice tmp_lat;
      fst::Reverse(*out_lat, &tmp_lat);
      fst::Determinize(tmp_lat, out_lat);
      fst::Reverse(*out_lat, &tmp_lat);
      fst::Determinize(tmp_lat, out_lat);
      fst::RmEpsilon(out_lat);
    }
  }

  fst::TopSort(out_lat);    
  std::vector<int32> state_times_tmp;
  KALDI_ASSERT(LatticeStateTimes(*out_lat, &state_times_tmp) == end_frame - begin_frame);

  // // Check if alpha scores before and after splitting are the same
  //LatticeInfo out_scores;
  //ComputeLatticeScores(*out_lat, &out_scores);
  //  KALDI_ASSERT(kaldi::ApproxEqual(out_scores.alpha_p[0], scores.alpha_p[begin_state - 1], .1));
  //  KALDI_ASSERT(kaldi::ApproxEqual(out_scores.beta_p[0], scores.beta_p[begin_state - 1] - scores.beta_p[0], .1));
  //for (size_t n_part = 1; n_part < out_scores.alpha_p.size()-1; n_part++) {
  //  KALDI_ASSERT(kaldi::ApproxEqual(out_scores.alpha_p[n_part], scores.alpha_p[n_part + begin_state - 1] - scores.beta_p[0], .1));
  //  KALDI_ASSERT(kaldi::ApproxEqual(out_scores.beta_p[n_part], scores.beta_p[n_part + begin_state - 1], .1));
  //}

  // Remove the acoustic scale that was previously added
  if (config_.supervision_config.acoustic_scale != 1.0) {
    fst::ScaleLattice(fst::AcousticLatticeScale(1 / config_.supervision_config.acoustic_scale), out_lat);
  }
}

void DiscriminativeSupervisionSplitter::PrepareLattice(
    Lattice *lat, LatticeInfo *scores) const {
  // Scale the lattice to appropriate acoustic scale. It is important to 
  // ensure this is equal to the acoustic scale used while training. This is 
  // because, on splitting lattices, the initial and final costs are added 
  // into the graph cost.
  KALDI_ASSERT(config_.supervision_config.acoustic_scale != 0.0);
  if (config_.supervision_config.acoustic_scale != 1.0)
    fst::ScaleLattice(fst::AcousticLatticeScale(config_.supervision_config.acoustic_scale), lat);

  LatticeStateTimes(*lat, &(scores->state_times));
  int32 num_states = lat->NumStates();
  std::vector<int32> inv_state_order(num_states);
  for (int32 s = 0; s < num_states; s++) {
    inv_state_order[s] = s;
  }

  // Order the states based on the state times. This is stronger than just
  // topological sort. This is required by the lattice splitting code.
  std::stable_sort(inv_state_order.begin(), 
                   inv_state_order.end(), 
                   OtherStlVectorComparator<int32>(scores->state_times));
  
  std::vector<int32> state_order(num_states);
  for (int32 s = 0; s < num_states; s++) {
    state_order[inv_state_order[s]] = s;
  }

  fst::StateSort(lat, state_order);
  ComputeLatticeScores(*lat, scores);
}

void DiscriminativeSupervisionSplitter::ComputeLatticeScores(const Lattice &lat,
    LatticeInfo *scores) const {
  LatticeStateTimes(lat, &(scores->state_times));
  ComputeLatticeAlphasAndBetas(lat, false, &(scores->alpha_p), &(scores->beta_p));
  scores->Check();  // This check will fail if the lattice is not breadth-first search sorted
}

void AppendSupervision(const std::vector<const DiscriminativeSupervision*> &input,
                       bool compactify,
                       std::vector<DiscriminativeSupervision> *output_supervision) {
  KALDI_ASSERT(!input.empty());
  int32 num_inputs = input.size();
  if (num_inputs == 1) {
    output_supervision->resize(1);
    (*output_supervision)[0] = *(input[0]);
    return;
  }
  output_supervision->clear();
  output_supervision->reserve(input.size());
  for (int32 i = 0; i < input.size(); i++) {
    const DiscriminativeSupervision &src = *(input[i]);
    KALDI_ASSERT(src.num_sequences == 1);
    if (compactify && !output_supervision->empty() &&
        output_supervision->back().weight == src.weight &&
        output_supervision->back().frames_per_sequence ==
        src.frames_per_sequence) {
      // Combine with current output
      // append src.den_lat to output_supervision->den_lat.
      AppendLattice(&output_supervision->back().den_lat, src.den_lat);
      if (i > 0) 
        KALDI_ASSERT((*output_supervision)[0].num_lat_present == src.num_lat_present);
      else
        output_supervision->back().num_lat_present = src.num_lat_present;
      if (src.num_lat_present)
        AppendLattice(&output_supervision->back().num_lat, src.num_lat);

      output_supervision->back().num_ali.insert(output_supervision->back().num_ali.end(), src.num_ali.begin(), src.num_ali.end());
      if (output_supervision->back().oracle_ali.size() > 0)
        output_supervision->back().oracle_ali.insert(output_supervision->back().oracle_ali.end(), src.oracle_ali.begin(), src.oracle_ali.end());
      if (output_supervision->back().weights.size() > 0)
        output_supervision->back().weights.insert(output_supervision->back().weights.end(), src.weights.begin(), src.weights.end());
      output_supervision->back().num_sequences++;
    } else {
      output_supervision->resize(output_supervision->size() + 1);
      output_supervision->back() = src;
    }

    output_supervision->back().Check();
  }
}

void AppendLattice(Lattice *lat, const Lattice &src_lat) {
  typedef Lattice::Arc Arc;
  typedef Arc::StateId StateId;

  std::vector<int32> state_times;
  int32 num_frames = LatticeStateTimes(*lat, &state_times);
  
  std::vector<int32> state_times_src;
  int32 num_frames_src = LatticeStateTimes(src_lat, &state_times_src);

  //Lattice check_lat= *lat;
  fst::Concat(lat, src_lat);
  fst::TopSort(lat);

  return;

  int32 num_states_orig = lat->NumStates();
  int32 num_states = num_states_orig;
  
  StateId src_start_state = src_lat.Start();
  KALDI_ASSERT(src_start_state == 0);

  lat->AddState(); num_states++;

  for (StateId s = 0; s < num_states_orig; s++) {
    LatticeWeight f = lat->Final(s);
    if (f != LatticeWeight::Zero()) {
      KALDI_ASSERT(state_times[s] == num_frames);
      lat->AddArc(s, Arc(0, 0, f, num_states_orig));
      lat->SetFinal(s, LatticeWeight::Zero());
    }
  }
  
  for (StateId s = 0; s < src_lat.NumStates(); s++) {
    if (s != src_start_state) {
      lat->AddState();
      num_states++;
    }
    StateId state_id = num_states_orig + s;
    KALDI_ASSERT(state_id == num_states - 1 && num_states == lat->NumStates());

    for (fst::ArcIterator<Lattice> aiter(src_lat, s); 
          !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      arc.nextstate += num_states_orig;
      lat->AddArc(state_id, arc);
    }
    
    LatticeWeight final_weight = src_lat.Final(s);
    lat->SetFinal(state_id, final_weight);
  }

  KALDI_ASSERT(lat->NumStates() == num_states);
  KALDI_ASSERT(num_states == num_states_orig + src_lat.NumStates());
  

  uint64 props = lat->Properties(fst::kTopSorted, true);
  lat->SetProperties(props, fst::kTopSorted);
  
  std::vector<int32> state_times_out;
  int32 num_frames_out = LatticeStateTimes(*lat, &state_times_out);
  KALDI_ASSERT(num_frames_out == num_frames + num_frames_src);

  for (StateId s = 0; s < lat->NumStates(); s++) {
    LatticeWeight f = lat->Final(s);
    if (f != LatticeWeight::Zero()) {
      KALDI_ASSERT(state_times_out[s] == num_frames_out &&
                   "Lattice is inconsistent (final-prob not at max_time)");
    }
    for (fst::ArcIterator<Lattice> aiter(*lat, s);
        !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0)
        KALDI_ASSERT(state_times_out[arc.nextstate] == state_times_out[s] + 1);
    }
  }
}

} // namespace discriminative 
} // namespace kaldi
