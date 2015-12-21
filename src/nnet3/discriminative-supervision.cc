
// called from MergeDiscriminativeExamples, this function merges the Supervision
// objects into one.  Requires (and checks) that they all have the same name.
static void MergeSupervision(
    const std::vector<const NnetChainSupervision*> &inputs,
    NnetChainSupervision *output) {
  int32 num_inputs = inputs.size(),
      num_indexes = 0;
  for (int32 n = 0; n < num_inputs; n++) {
    KALDI_ASSERT(inputs[n]->name == inputs[0]->name);
    num_indexes += inputs[n]->indexes.size();
  }
  output->name = inputs[0]->name;
  std::vector<const chain::Supervision*> input_supervision;
  input_supervision.reserve(inputs.size());
  for (int32 n = 0; n < num_inputs; n++)
    input_supervision.push_back(&(inputs[n]->supervision));
  std::vector<chain::Supervision> output_supervision;
  bool compactify = true;
  AppendSupervision(input_supervision,
                         compactify,
                         &output_supervision);
  if (output_supervision.size() != 1)
    KALDI_ERR << "Failed to merge 'chain' examples-- inconsistent lengths "
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
      KALDI_ASSERT(iter->n == 0 && "Merging already-merged chain egs");
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
  output->CheckDim();
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
  LatticeStateTimes(den_lat, &frame_);

  int32 num_states = den_lat.NumStates(),
        num_frames = supervision_.frames_per_sequence * supervision_.num_sequences;
  KALDI_ASSERT(num_states > 0);
  int32 start_state = den_lat.Start();
  // Lattice should be top-sorted and connected, so start-state must be 0.
  KALDI_ASSERT(start_state == 0 && "Expecting start-state to be 0");
  KALDI_ASSERT(frame_[start_state] == 0);
}

void DiscriminativeSupervisionSplitter::GetFrameRange(int32 begin_frame, int32 num_frames,
                                        Supervision *out_supervision) const {
  int32 end_frame = begin_frame + num_frames;
  // Note: end_frame is not included in the range of frames that the
  // output supervision object covers; it's one past the end.
  KALDI_ASSERT(num_frames > 0 && begin_frame >= 0 &&
               begin_frame + num_frames <=
               supervision_.num_sequences * supervision_.frames_per_sequence);
  std::vector<int32>::const_iterator begin_iter =
      std::lower_bound(frame_.begin(), frame_.end(), begin_frame),
      end_iter = std::lower_bound(begin_iter, frame_.end(), end_frame);
  KALDI_ASSERT(*begin_iter == begin_frame &&
               (begin_iter == frame_.begin() || begin_iter[-1] < begin_frame));
  // even if end_frame == supervision_.num_frames, there should be a state with
  // that frame index.
  KALDI_ASSERT(end_iter[-1] < end_frame &&
               (end_iter < frame_.end() || *end_iter == end_frame));
  int32 begin_state = begin_iter - frame_.begin(),
      end_state = end_iter - frame_.begin();

  Lattice den_lat;
  ConverLattice(supervision_.den_lat, &den_lat);
  Lattice out_den_lat;

  CreateRangeLattice(den_lat,
                     begin_frame, end_frame,
                     begin_state, end_state, 
                     &out_den_lat);

  Lattice out_num_lat;
  if (supervision_.num_lat_present) {
    Lattice num_lat;
    ConvertLattice(supervision_.num_lat, &num_lat);
    
    CreateRangeLattice(num_lat, 
                       begin_frame, end_frame,
                       begin_state, end_state, 
                       &out_num_lat);
  }

  KALDI_ASSERT(out_supervision->fst.NumStates() > 0);
  KALDI_ASSERT(supervision_.num_sequences == 1);

  out_supervision->den_lat.CopyFrom(out_den_lat);
  out_supervision->num_lat_present = supervision_.num_lat_present;
  if (supervision_.num_lat_present)
    out_supervision->num_lat.CopyFrom(out_num_lat);
  out_supervision->num_sequences = 1;
  out_supervision->weight = supervision_.weight;
  out_supervision->frames_per_sequence = num_frames;
  out_supervision->label_dim = supervision_.label_dim;
}

void DiscriminativeSupervisionSplitter::CreateRangeLattice(
    const Lattice &in_lat,
    const std::vector<int32> &state_times,
    int32 begin_frame, int32 end_frame,
    int32 begin_state, int32 end_state,
    Lattice *out_lat) const {
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
      // from our actual initial state.
      out_lat->AddArc(start_state, 
          LatticeArc(0, 0, LatticeWeight::One(), output_state));
    } else {
      KALDI_ASSERT(state_times_[state] < end_frame);
    }
    for (fst::ArcIterator<Lattice> aiter(in_lat, state); 
          !aiter.Done(); aiter.Next()) {
      const LatticeArc &arc = aiter.Value();
      int32 nextstate = arc.nextstate;
      if (nextstate >= end_state) {
        // A transition to any state outside the range becomes a transition to
        // our special final-state.
        out_lat->AddArc(output_state,
            LatticeArc(arc.ilabel, arc.olabel, arc.weight, final_state));
      } else {
        int32 output_nextstate = arc.nextstate - begin_state + 1;
        out_lat->AddArc(output_state,
            LatticeArc(arc.ilabel, arc.olabel, arc.weight, output_nextstate));
      }
    }
  }
}
