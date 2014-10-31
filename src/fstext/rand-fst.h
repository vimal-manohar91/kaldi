// fstext/rand-fst.h

// Copyright 2009-2011  Microsoft Corporation

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

#ifndef KALDI_FSTEXT_RAND_FST_H_
#define KALDI_FSTEXT_RAND_FST_H_

#include <sstream>
#include <string>

#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include "base/kaldi-math.h"
#include "util/common-utils.h"

namespace fst {

// Note: all weights are constructed from nonnegative floats.
// (so no "negative costs").
struct RandFstOptions {
  int32 n_syms;
  int32 n_states;
  int32 n_arcs;
  int32 n_final;
  bool allow_empty;
  bool acyclic;
  float weight_multiplier;
  bool uniq_labels;
  bool same_iolabels;
  
  RandFstOptions() {  // Initializes the options randomly.
    n_syms = 2 + kaldi::Rand() % 5;
    n_states = 3 + kaldi::Rand() % 10;
    n_arcs = 5 + kaldi::Rand() % 30;
    n_final = 1 + kaldi::Rand() % 3;
    allow_empty = true;
    acyclic = false;
    weight_multiplier = 0.25;
    uniq_labels = false;
    same_iolabels = false;
  }

  void Register(kaldi::OptionsItf *po) {
    po->Register("num-syms", &n_syms,
        "Number of allowed symbols");
    po->Register("num-states", &n_states,
        "Number of states in FST");
    po->Register("num-arcs", &n_arcs,
        "Number of arcs in FST");
    po->Register("num-final", &n_final,
        "Number of final statees");
    po->Register("allow-empty", &allow_empty,
        "");
    po->Register("acyclic", &acyclic, "Create acyclic FSTs");
    po->Register("weight-multiplier", &weight_multiplier, 
        "The weights are all multiples of this.");
    po->Register("uniq-labels", &uniq_labels,
        "Make the arc labels unique; "
        "input and output labels are forced to be the same.\n"
        "Applicable only to timed FST.");
    po->Register("same-iolabels", &same_iolabels, 
        "Force input and output labels to the same.\n"
        "Applicable only to timed FST.");
  }
};


/// Returns a random FST.  Useful for randomized algorithm testing.
/// Only works if weight can be constructed from float.
template<class Arc> VectorFst<Arc>* RandFst(RandFstOptions opts = RandFstOptions() ) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  VectorFst<Arc> *fst = new VectorFst<Arc>();

 start:

  // Create states.
  vector<StateId> all_states;
  for (size_t i = 0;i < (size_t)opts.n_states;i++) {
    StateId this_state = fst->AddState();
    if (i == 0) fst->SetStart(i);
    all_states.push_back(this_state);
  }
  // Set final states.
  for (size_t j = 0;j < (size_t)opts.n_final;j++) {
    StateId id = all_states[kaldi::Rand() % opts.n_states];
    Weight weight = (Weight)(opts.weight_multiplier*(kaldi::Rand() % 5));
    fst->SetFinal(id, weight);
  }
  // Create arcs.
  for (size_t i = 0;i < (size_t)opts.n_arcs;i++) {
    Arc a;
    StateId start_state;
    if(!opts.acyclic) { // no restriction on arcs.
      start_state = all_states[kaldi::Rand() % opts.n_states];
      a.nextstate = all_states[kaldi::Rand() % opts.n_states];
    } else {
      start_state = all_states[kaldi::Rand() % (opts.n_states-1)];
      a.nextstate = start_state + 1 + (kaldi::Rand() % (opts.n_states-start_state-1));
    }
    a.ilabel = kaldi::Rand() % opts.n_syms;
    a.olabel = kaldi::Rand() % opts.n_syms;  // same input+output vocab.
    a.weight = (Weight) (opts.weight_multiplier*(kaldi::Rand() % 4));
    
    fst->AddArc(start_state, a);
  }

  // Trim resulting FST.
  Connect(fst);
  if (opts.acyclic)
    assert(fst->Properties(kAcyclic, true) & kAcyclic);
  if (fst->Start() == kNoStateId && !opts.allow_empty) {
    goto start;
  }
  return fst;
}


/// Returns a random FST.  Useful for randomized algorithm testing.
/// Only works if weight can be constructed from a pair of floats
template<class Arc> VectorFst<Arc>* RandPairFst(RandFstOptions opts = RandFstOptions() ) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  VectorFst<Arc> *fst = new VectorFst<Arc>();

 start:

  // Create states.
  vector<StateId> all_states;
  for (size_t i = 0;i < (size_t)opts.n_states;i++) {
    StateId this_state = fst->AddState();
    if (i == 0) fst->SetStart(i);
    all_states.push_back(this_state);
  }
  // Set final states.
  for (size_t j = 0; j < (size_t)opts.n_final;j++) {
    StateId id = all_states[kaldi::Rand() % opts.n_states];
    Weight weight (opts.weight_multiplier*(kaldi::Rand() % 5), opts.weight_multiplier*(kaldi::Rand() % 5));
    fst->SetFinal(id, weight);
  }
  // Create arcs.
  for (size_t i = 0;i < (size_t)opts.n_arcs;i++) {
    Arc a;
    StateId start_state;
    if(!opts.acyclic) { // no restriction on arcs.
      start_state = all_states[kaldi::Rand() % opts.n_states];
      a.nextstate = all_states[kaldi::Rand() % opts.n_states];
    } else {
      start_state = all_states[kaldi::Rand() % (opts.n_states-1)];
      a.nextstate = start_state + 1 + (kaldi::Rand() % (opts.n_states-start_state-1));
    }
    a.ilabel = kaldi::Rand() % opts.n_syms;
    a.olabel = kaldi::Rand() % opts.n_syms;  // same input+output vocab.
    a.weight = Weight (opts.weight_multiplier*(kaldi::Rand() % 4), opts.weight_multiplier*(kaldi::Rand() % 4));
    
    fst->AddArc(start_state, a);
  }

  // Trim resulting FST.
  Connect(fst);
  if (opts.acyclic)
    assert(fst->Properties(kAcyclic, true) & kAcyclic);
  if (fst->Start() == kNoStateId && !opts.allow_empty) {
    goto start;
  }
  return fst;
}


/// Returns a random timed FST.  Useful for randomized algorithm testing.
/// Only works if weight can be constructed from a pair of floats
/// This is different from the previous function because this allows only
/// certain arcs that fulfil the property that the distance from the start 
/// state to a particular state using any arc would be the same. That is
/// the FST has an inbuilt notion of time.
template<class Arc> VectorFst<Arc>* RandPairTimedFst(RandFstOptions opts = RandFstOptions() ) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  VectorFst<Arc> *fst = new VectorFst<Arc>();

 start:

  // Create states.
  vector<StateId> all_states;

  int32 max_time = 1 + (kaldi::Rand() % (opts.n_states-1));
  int32 n_states = 0;

  // Vectors to store the times corresponding to each state and the states
  // at each time
  vector<int32> state_times;
  vector<vector<StateId> > time_states;
  time_states.resize(max_time + 1);

  // Create atleast one state for each time
  for (int32 t = 0; t <= max_time; t++, n_states++) {
    StateId this_state = fst->AddState();
    if (t == 0) fst->SetStart(0);
    all_states.push_back(this_state);
    state_times.push_back(t);
    time_states[t].push_back(n_states);
  }

  // Create paths.
  for (size_t i = 0; i < (size_t)opts.n_arcs;) {
    for (int32 t = 0, s = 0, e = 0; t < max_time; t++) {
      StateId start_state;

      // Choose a start state for the arc starting at time t
      if (t == 0) {
        s = kaldi::Rand() % (time_states[t].size());
        start_state = all_states[time_states[t][s]];
      } else {
        start_state = all_states[time_states[t][e]];
      }
      
      Arc a;
      // Choose an end state for the arc. Either choose one of the existing
      // states or create a new one if the total number of states is still less
      // than opts.n_states. Also ensure we do not exceed the maximum number of
      // final states allowed.
      {
        if (n_states < opts.n_states && 
            (t+1 < max_time || time_states[t+1].size() < opts.n_final) ) {
          e = kaldi::Rand() % (time_states[t+1].size() + 1);
        } else {
          e = kaldi::Rand() % (time_states[t+1].size());
        }
        
        if (e >= time_states[t+1].size()) {
          KALDI_ASSERT(e == time_states[t+1].size());
          StateId this_state = fst->AddState();
          all_states.push_back(this_state);
          state_times.push_back(t+1);
          time_states[t+1].push_back(n_states++);
        }
        a.nextstate = all_states[time_states[t+1][e]];
      }
    
      if (opts.uniq_labels) {
        a.ilabel = i + 1;
        a.olabel = i + 1;
      } else {
        a.ilabel = 1 + kaldi::Rand() % opts.n_syms;
        if (opts.same_iolabels) {
          a.olabel = a.ilabel;
        } else {
          a.olabel = 1 + kaldi::Rand() % opts.n_syms;  // same input+output vocab.
        }
      }

      a.weight = Weight (opts.weight_multiplier*(kaldi::Rand() % 4), opts.weight_multiplier*(kaldi::Rand() % 4));
    
      fst->AddArc(start_state, a);
      i++;
    }
  }
  
  // Set final states.
  for (size_t j = 0; j < (size_t) time_states[max_time].size();j++) {
    StateId id = all_states[time_states[max_time][j]];
    Weight weight (opts.weight_multiplier*(kaldi::Rand() % 5), opts.weight_multiplier*(kaldi::Rand() % 5));
    fst->SetFinal(id, weight);
  }

  // Trim resulting FST.
  Connect(fst);
  assert(fst->Properties(kAcyclic, true) & kAcyclic);
  if (fst->Start() == kNoStateId && !opts.allow_empty) {
    goto start;
  }
  return fst;
}

} // end namespace fst.


#endif

