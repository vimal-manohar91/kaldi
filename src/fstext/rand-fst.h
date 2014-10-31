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

namespace fst {

// Note: all weights are constructed from nonnegative floats.
// (so no "negative costs").
struct RandFstOptions {
  size_t n_syms;
  size_t n_states;
  size_t n_arcs;
  size_t n_final;
  bool allow_empty;
  bool acyclic;
  float weight_multiplier;
  RandFstOptions() {  // Initializes the options randomly.
    n_syms = 2 + kaldi::Rand() % 5;
    n_states = 3 + kaldi::Rand() % 10;
    n_arcs = 5 + kaldi::Rand() % 30;
    n_final = 1 + kaldi::Rand()%3;
    allow_empty = true;
    acyclic = false;
    weight_multiplier = 0.25;
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
    
      a.ilabel = 1 + kaldi::Rand() % opts.n_syms;
      a.olabel = 1 + kaldi::Rand() % opts.n_syms;  // same input+output vocab.
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

