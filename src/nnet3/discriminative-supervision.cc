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

DiscriminativeSupervision::DiscriminativeSupervision(const DiscriminativeSupervision &other):
    weight(other.weight), num_sequences(other.num_sequences),
    frames_per_sequence(other.frames_per_sequence), label_dim(other.label_dim),
    num_ali(other.num_ali), oracle_ali(other.oracle_ali),
    weights(other.weights),
    num_lat_present(other.num_lat_present),
    num_lat(other.num_lat),
    den_lat(other.den_lat) { }

void DiscriminativeSupervision::Swap(DiscriminativeSupervision *other) {
  std::swap(weight, other->weight);
  std::swap(num_sequences, other->num_sequences);
  std::swap(frames_per_sequence, other->frames_per_sequence);
  std::swap(label_dim, other->label_dim);
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
      label_dim == other.label_dim &&
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
  WriteToken(os, binary, "<LabelDim>");
  WriteBasicType(os, binary, label_dim);
  KALDI_ASSERT(frames_per_sequence > 0 && label_dim > 0 &&
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
  ExpectToken(is, binary, "<LabelDim>");
  ReadBasicType(is, binary, &label_dim);
  KALDI_ASSERT(frames_per_sequence > 0 && label_dim > 0 &&
               num_sequences > 0);
  
  ExpectToken(is, binary, "<NumAli>");
  ReadIntegerVector(is, binary, &num_ali);

  std::string token;
  ReadToken(is, binary, &token);

  if (token == "<NumLat>") {
    num_lat_present = true;
    CompactLattice *clat;
    if (!ReadCompactLattice(is, binary, &clat) || clat == NULL) {
      // We can't return error status from this function so we
      // throw an exception. 
      KALDI_ERR << "Error reading CompactLattice from stream";
    }
    ConvertLattice(*clat, &num_lat);
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
    CompactLattice *clat;
    if (!ReadCompactLattice(is, binary, &clat) || clat == NULL) {
      // We can't return error status from this function so we
      // throw an exception. 
      KALDI_ERR << "Error reading CompactLattice from stream";
    }
    ConvertLattice(*clat, &den_lat);
  }

  ExpectToken(is, binary, "</DiscriminativeSupervision>");
}

bool LatticeToDiscrminativeSupervision(const std::vector<int32> &num_ali,
                                       const Lattice &num_lat, 
                                       const Lattice &den_lat,
                                       BaseFloat weight,
                                       DiscriminativeSupervision *supervision,
                                       const std::vector<BaseFloat> *weights,
                                       const std::vector<int32> *oracle_alignment) {
  supervision->weight = weight;
  supervision->num_sequences = 1;
  supervision->frames_per_sequence = num_ali.size();
  supervision->num_ali = num_ali;
  supervision->num_lat_present = true;
  supervision->num_lat = num_lat;
  supervision->den_lat = den_lat;
  if (weights)
    supervision->weights = *weights;
  if (oracle_alignment)
    supervision->oracle_ali = *oracle_alignment;

  supervision->Check();

  return true;
}

bool LatticeToDiscriminativeSupervision(const std::vector<int32> &num_ali,
                                        const Lattice &den_lat, 
                                        BaseFloat weight,
                                        DiscriminativeSupervision *supervision,
                                        const std::vector<BaseFloat> *weights,
                                        const std::vector<int32> *oracle_alignment) {
  supervision->weight = weight;
  supervision->num_sequences = 1;
  supervision->frames_per_sequence = num_ali.size();
  supervision->num_ali = num_ali;
  supervision->num_lat_present = false;
  supervision->den_lat = den_lat;
  if (weights)
    supervision->weights = *weights;
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
    const DiscriminativeSupervision &supervision):
    supervision_(supervision) {
  if (supervision_.num_sequences != 1) {
    KALDI_WARN << "Splitting already-reattached sequence (only expected in "
               << "testing code)";
  }
  const Lattice &den_lat = supervision_.den_lat;;
  ComputeLatticeScores(den_lat, &den_lat_scores_);
  
  if (supervision_.num_lat_present) {
    const Lattice &num_lat = supervision_.num_lat;;
    ComputeLatticeScores(num_lat, &num_lat_scores_);
  }
  
  int32 num_states = den_lat.NumStates(),
        num_frames = supervision_.frames_per_sequence * supervision_.num_sequences;

  KALDI_ASSERT(num_states > 0);
  int32 start_state = den_lat.Start();
  // Lattice should be top-sorted and connected, so start-state must be 0.
  KALDI_ASSERT(start_state == 0 && "Expecting start-state to be 0");
  KALDI_ASSERT(num_lat_scores_.state_times[start_state] == 0);

  KALDI_ASSERT(num_frames == num_lat_scores_.state_times.size());
  KALDI_ASSERT(num_frames == den_lat_scores_.state_times.size());
}

void DiscriminativeSupervisionSplitter::GetFrameRange(int32 begin_frame, int32 num_frames,
                                        DiscriminativeSupervision *out_supervision) const {
  int32 end_frame = begin_frame + num_frames;
  // Note: end_frame is not included in the range of frames that the
  // output supervision object covers; it's one past the end.
  KALDI_ASSERT(num_frames > 0 && begin_frame >= 0 &&
               begin_frame + num_frames <=
               supervision_.num_sequences * supervision_.frames_per_sequence);

  CreateRangeLattice(den_lat_,
                     den_lat_scores_,
                     begin_frame, end_frame,
                     &(out_supervision->den_lat));

  if (num_lat_present_) {
    CreateRangeLattice(num_lat_, 
                       num_lat_scores_,
                       begin_frame, end_frame,
                       &(out_supervision->num_lat));
  }
  out_supervision->num_lat_present = num_lat_present_;

  KALDI_ASSERT(supervision_.num_sequences == 1);

  out_supervision->num_ali.clear();
  out_supervision->num_ali.push_back(1);     // dummy to align with the lattice
  std::copy(supervision_.num_ali.begin() + begin_frame,
      supervision_.num_ali.begin() + end_frame,
      std::back_inserter(out_supervision->num_ali));
  
  out_supervision->oracle_ali.clear();
  if (supervision_.oracle_ali.size() > 0) {
    out_supervision->oracle_ali.push_back(1);  // dummy to align with the lattice
    std::copy(supervision_.oracle_ali.begin() + begin_frame,
        supervision_.oracle_ali.begin() + end_frame,
        std::back_inserter(out_supervision->oracle_ali));
  }

  out_supervision->weights.clear();
  if (supervision_.weights.size() > 0) {
    out_supervision->weights.push_back(0.0);     // dummy to align with the lattice
    std::copy(supervision_.weights.begin() + begin_frame,
        supervision_.weights.begin() + end_frame,
        std::back_inserter(out_supervision->weights));
  }

  out_supervision->num_sequences = 1;
  out_supervision->weight = supervision_.weight;
  out_supervision->frames_per_sequence = num_frames;
  out_supervision->label_dim = supervision_.label_dim;

  out_supervision->Check();
}

void DiscriminativeSupervisionSplitter::CreateRangeLattice(
    const Lattice &in_lat, const LatticeInfo &scores,
    int32 begin_frame, int32 end_frame,
    Lattice *out_lat) const {
  const std::vector<int32> &state_times = scores.state_times;
  
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
  int32 begin_state = begin_iter - state_times.begin(),
          end_state = end_iter - state_times.begin();

  KALDI_ASSERT(end_state > begin_state);
  out_lat->DeleteStates();
  out_lat->ReserveStates(end_state - begin_state + 2);
  int32 start_state = out_lat->AddState();
  out_lat->SetStart(start_state);
  for (int32 i = begin_state; i < end_state; i++)
    out_lat->AddState();
  // Add the special final-state.
  int32 final_state = out_lat->AddState();
  out_lat->SetFinal(final_state, LatticeWeight::One());
  for (int32 state = begin_state; state < end_state; state++) {
    int32 output_state = state - begin_state + 1;
    if (state_times[state] == begin_frame) {
      // we'd like to make this an initial state, but OpenFst doesn't allow
      // multiple initial states.  Instead we add an epsilon transition to it
      // from our actual initial state
      LatticeWeight weight = LatticeWeight::One();
      weight.SetValue1(scores.alpha_p[output_state]);

      out_lat->AddArc(start_state, 
                      LatticeArc(0, 0, weight, output_state));
    } else {
      KALDI_ASSERT(scores.state_times[state] < end_frame);
    }
    for (fst::ArcIterator<Lattice> aiter(in_lat, state); 
          !aiter.Done(); aiter.Next()) {
      const LatticeArc &arc = aiter.Value();
      int32 nextstate = arc.nextstate;
      if (nextstate >= end_state) {
        // A transition to any state outside the range becomes a transition to
        // our special final-state. The weight is just the backward probability.
        LatticeWeight weight = LatticeWeight::One();
        weight.SetValue1(arc.weight.Value1() + scores.beta_p[nextstate]);

        // LatticeWeight weight = arc.weight;
        // weight.SetValue1(arc.weight.Weight().Value1() + scores.beta_p[state]);

        out_lat->AddArc(output_state,
            LatticeArc(arc.ilabel, arc.olabel, weight, final_state));
      } else {
        int32 output_nextstate = arc.nextstate - begin_state + 1;
        out_lat->AddArc(output_state,
            LatticeArc(arc.ilabel, arc.olabel, arc.weight, output_nextstate));
      }
    }
  }
}

void DiscriminativeSupervisionSplitter::ComputeLatticeScores(const Lattice &lat,
    LatticeInfo *scores) const {
  LatticeStateTimes(lat, &(scores->state_times));
  ComputeLatticeAlphasAndBetas(lat, false, &(scores->alpha_p), &(scores->beta_p));
  scores->Check();
}

void AppendSupervision(const std::vector<const DiscriminativeSupervision*> &input,
                       bool compactify,
                       std::vector<DiscriminativeSupervision> *output_supervision) {
  KALDI_ASSERT(!input.empty());
  int32 label_dim = input[0]->label_dim,
      num_inputs = input.size();
  if (num_inputs == 1) {
    output_supervision->resize(1);
    (*output_supervision)[0] = *(input[0]);
    return;
  }
  for (int32 i = 1; i < num_inputs; i++)
    KALDI_ASSERT(input[i]->label_dim == label_dim &&
                 "Trying to append incompatible DiscriminativeSupervision objects");
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
  int32 num_frames_src = LatticeStateTimes(src_lat, &state_times);
  int32 num_frames = LatticeStateTimes(*lat, &state_times);

  fst::Concat(lat, src_lat);

  KALDI_ASSERT(LatticeStateTimes(*lat, &state_times) == num_frames + num_frames_src);

  /*
  int32 num_states_orig = lat->NumStates();
  int32 num_states = num_states_orig;

  for (StateId s = 0; s < num_states_orig; s++) {
    if (state_times[s] == num_frames) {
      for (fst::ArcIterator<Lattice> aiter(src_lat, 0);
            !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        int32 state_id = num_states_orig + arc.nextstate - 1;

        // Add the final weight of the first lattice into the arcs that go 
        // to the second lattice
        LatticeWeight weight;
        weight.SetValue1(arc.weight.Value1() + lat->Final(s).Value1());
        weight.SetValue2(arc.weight.Value2() + lat->Final(s).Value2());

        lat->AddArc(s, Arc(arc.ilabel, arc.olabel, weight, state_id));
      }
      lat->SetFinal(s, LatticeWeight::Zero());
    }
  }
  
  for (StateId s = 1; s < src_lat.NumStates(); s++) {
    lat->AddState();
    num_states++;
    int32 state_id = num_states_orig + s - 1;
    KALDI_ASSERT(state_id == num_states - 1 && num_states == lat->NumStates());

    for (fst::ArcIterator<Lattice> aiter(src_lat, s); 
          !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      arc.nextstate += num_states_orig - 1;
      lat->AddArc(state_id, arc);
    }
    
    LatticeWeight final_weight = src_lat.Final(s);
    lat->SetFinal(state_id, final_weight);
  }

  KALDI_ASSERT(lat->NumStates() == num_states);
  KALDI_ASSERT(lat->Properties(fst::kTopSorted, true) == 0
      && "Input lattice must be topologically sorted.");

  int32 num_frames_out = LatticeStateTimes(*lat, &state_times);
  KALDI_ASSERT(num_frames_out == num_frames + num_frames_src);

  for (StateId s = 0; s < lat->NumStates(); s++) {
    LatticeWeight f = lat->Final(s);
    if (f != LatticeWeight::Zero()) {
      KALDI_ASSERT(state_times[s] == num_frames_out &&
                   "Lattice is inconsistent (final-prob not at max_time)");
    }
    for (fst::ArcIterator<Lattice> aiter(*lat, s);
        !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      KALDI_ASSERT(state_times[arc.nextstate] == state_times[s] + 1);
    }
  }
  */
}

} // namespace discriminative 
} // namespace kaldi
