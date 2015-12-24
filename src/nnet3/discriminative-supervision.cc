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

#include "lat/lattice-functions.h"
#include "nnet3/discriminative-supervision.h"

namespace kaldi {
namespace nnet3 {

bool LatticeToDiscrminativeSupervision(const std::vector<int32> &num_ali,
                                       const CompactLattice &num_lat, 
                                       const CompactLattice &den_lat,
                                       BaseFloat weight,
                                       DiscriminativeSupervision *supervision) {
  supervision->weight = weight;
  supervision->num_sequencues = 1;
  supervision->frames_per_sequencue = num_ali.size();
  supervision->num_ali = num_ali;
  supervision->num_lat_present = true;
  supervision->num_lat = num_lat;
  supervision->den_lat = den_lat;

  Check();

  return true;
}

bool LatticeToDiscrminativeSupervision(const std::vector<int32> &num_ali,
                                       const CompactLattice &num_lat, 
                                       BaseFloat weight,
                                       DiscriminativeSupervision *supervision) {
  supervision->weight = weight;
  supervision->num_sequencues = 1;
  supervision->frames_per_sequencue = num_ali.size();
  supervision->num_ali = num_ali;
  supervision->num_lat_present = false;
  supervision->den_lat = den_lat;

  Check();

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
    int32 max_time = CompactLatticeStateTimes(den_lat, &state_times);
    KALDI_ASSERT(max_time == num_frames);
  }

  if (num_lat_present) {
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
  Lattice den_lat;
  ConvertLattice(supervision_.den_lat, &den_lat);
  ComputeLatticeScores(den_lat, &den_lat_scores_);
  
  if (supervision_.num_lat_present) {
    Lattice num_lat;
    ConvertLattice(supervision_.num_lat, &num_lat);
    ComputeLatticeScores(num_lat, &num_lat_scores_);
  }
  
  int32 num_states = den_lat.NumStates(),
        num_frames = supervision_.frames_per_sequence * supervision_.num_sequences;

  KALDI_ASSERT(num_states > 0);
  int32 start_state = den_lat.Start();
  // Lattice should be top-sorted and connected, so start-state must be 0.
  KALDI_ASSERT(start_state == 0 && "Expecting start-state to be 0");
  KALDI_ASSERT(num_lat_scores_.state_times.[start_state] == 0);

  KALDI_ASSERT(num_frames == num_lat_scores_.state_times.size());
  KALDI_ASSERT(num_frames == den_lat_scores_.state_times.size());
}

void DiscriminativeSupervisionSplitter::GetFrameRange(int32 begin_frame, int32 num_frames,
                                        Supervision *out_supervision) const {
  int32 end_frame = begin_frame + num_frames;
  // Note: end_frame is not included in the range of frames that the
  // output supervision object covers; it's one past the end.
  KALDI_ASSERT(num_frames > 0 && begin_frame >= 0 &&
               begin_frame + num_frames <=
               supervision_.num_sequences * supervision_.frames_per_sequence);

  Lattice out_den_lat;
  CreateRangeLattice(den_lat_,
                     den_lat_scores_.state_times,
                     begin_frame, end_frame,
                     &out_den_lat);

  Lattice out_num_lat;
  if (num_lat_present_) {
    CreateRangeLattice(num_lat_, 
                       num_lat_scores_,
                       begin_frame, end_frame,
                       &out_num_lat);
  }

  KALDI_ASSERT(supervision_.num_sequences == 1);

  out_supervision->den_lat.CopyFrom(out_den_lat);
  out_supervision->num_lat_present = num_lat_present_;
  if (num_lat_present_)
    out_supervision->num_lat.CopyFrom(out_num_lat);

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

  std::vector<int32> state_times;
  KALDI_ASSERT(num_frames == LatticeStateTimes(lat, &state_times));
  KALDI_ASSERT(num_frames == LatticeStateTimes(num_lat, &state_times));

  KALDI_ASSERT(static_cast<int32>(out_supervision->num_ali.size()) == num_frames);
  KALDI_ASSERT(out_supervision->oracle_ali.size() == 0 || static_cast<int32>(out_supervision->oracle_ali.size()) == num_frames);
  KALDI_ASSERT(out_supervision->weights.size() == 0 || static_cast<int32>(out_supervision->weights.size()) == num_frames);
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
      KALDI_ASSERT(state_times_[state] < end_frame);
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
  lat.ComputeLatticeAlphasAndBetas(lat, scores->alpha_p, scores->beta_p);
  scores->Check();
}

// called from MergeDiscriminativeExamples, this function merges the Supervision
// objects into one.  Requires (and checks) that they all have the same name.

static void MergeSupervision(
    const std::vector<const NnetDiscriminativeSupervision*> &inputs,
    NnetDiscriminativeSupervision *output) {
  int32 num_inputs = inputs.size(), num_indexes = 0;
  for (int32 n = 0; n < num_inputs; n++) {
    KALDI_ASSERT(inputs[n]->name == inputs[0]->name);
    num_indexes += inputs[n]->indexes.size();
  }
  output->name = inputs[0]->name;
  std::vector<const DiscriminativeSupervision*> input_supervision;
  input_supervision.reserve(inputs.size());
  for (int32 n = 0; n < num_inputs; n++)
    input_supervision.push_back(&(inputs[n]->supervision));
  std::vector<DiscriminativeSupervision> output_supervision;
  bool compactify = true;
  AppendSupervision(input_supervision,
                         compactify,
                         &output_supervision);
  if (output_supervision.size() != 1)
    KALDI_ERR << "Failed to merge 'sequence' examples-- inconsistent lengths "
              << "or weights?";
  output->supervision.Swap(&(output_supervision[0]));

  output->indexes.clear();
  output->indexes.reserve(num_indexes);
  for (int32 n = 0; n < num_inputs; n++) {
    const std::vector<Index> &src_indexes = inputs[n]->indexes;
    int32 cur_size = output->indexes.size();
    output->indexes.insert(output->indexes.end(),
                           src_indexes.begin(), src_indexes.end());
    std::vector<Index>::iterator iter = output->indexes.begin() + cur_size,
        end = output->indexes.end();
    // change the 'n' index to correspond to the index into 'input'.
    // Each example gets a different 'n' value, starting from 0.
    for (; iter != end; ++iter) {
      KALDI_ASSERT(iter->n == 0 && "Merging already-merged sequence egs");
      iter->n = n;
    }
  }
  KALDI_ASSERT(output->indexes.size() == num_indexes);
  // OK, at this point the 'indexes' will be in the wrong order,
  // because they should be first sorted by 't' and next by 'n'.
  // 'sort' will fix this, due to the operator < on type Index.
  std::sort(output->indexes.begin(), output->indexes.end());

  // merge the deriv_weights.
  if (inputs[0]->deriv_weights.Dim() != 0) {
    int32 frames_per_sequence = inputs[0]->deriv_weights.Dim();
    output->deriv_weights.Resize(output->indexes.size(), kUndefined);
    KALDI_ASSERT(output->deriv_weights.Dim() ==
                 frames_per_sequence * num_inputs);
    for (int32 n = 0; n < num_inputs; n++) {
      const Vector<BaseFloat> &src_deriv_weights = inputs[n]->deriv_weights;
      KALDI_ASSERT(src_deriv_weights.Dim() == frames_per_sequence);
      // the ordering of the deriv_weights corresponds to the ordering of the
      // Indexes, where the time dimension has the greater stride.
      for (int32 t = 0; t < frames_per_sequence; t++) {
        output->deriv_weights(t * num_inputs + n) = src_deriv_weights(t);
      }
    }
  }
  output->Check();
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
    const Supervision &src = *(input[i]);
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

    output_supervision.back().Check();
  }
}

static void AppendLattice(Lattice *lat, const Lattice &src_lat) {
  typedef Lattice::Arc Arc;
  typedef Arc::StateId StateId;

  std::vector<int32> state_times;
  int32 num_frames_src = LatticeStateTimes(src_lat, &state_times);
  int32 num_frames = LatticeStateTimes(*lat, &state_times);

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
        LatticeWeight weight = arc.weight.Weight();
        weight.SetValue1(weight.Value1() + lat->Final(s).Value1());
        weight.SetValue2(weight.Value2() + lat->Final(s).Value2());

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
      lat->Add(state_id, arc);
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
    Weight f = lat->Final(s);
    if (f != Weight::Zero()) {
      KALDI_ASSERT(state_times[s] == num_frames_out &&
                   "Lattice is inconsistent (final-prob not at max_time)");
    }
    for (fst::ArcIterator<Lattice> aiter(*lat, s);
        !aiter.Done(); aiter.Next()) {
      KALDI_ASSERT(state_times[aiter.nextstate] == state_times[s] + 1);
    }
  }
}

} // namespace nnet3
} // namespace kaldi
