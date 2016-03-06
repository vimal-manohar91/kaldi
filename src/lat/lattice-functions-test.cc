// lat/lattice-functions-test.cc

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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


#include "lat/kaldi-lattice.h"
#include "lat/minimize-lattice.h"
#include "lat/push-lattice.h"
#include "hmm/transition-model.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/lattice-functions.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "lat/lattice-functions.h"
#include "hmm/transition-model.h"
#include "util/stl-utils.h"
#include "base/kaldi-math.h"
#include "base/timer.h"

namespace kaldi {
using namespace fst;

struct RandLatticeOptions {
  int32 n_syms;
  int32 n_states;
  int32 n_arcs;
  int32 n_final;
  bool allow_empty;
  bool acyclic;
  float weight_multiplier;
  bool uniq_labels;
  bool same_iolabels;
  RandLatticeOptions() {  // Initializes the options randomly.
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

/// Returns a random timed FST.  Useful for randomized algorithm testing.
/// Only works if weight can be constructed from a pair of floats
/// This is different from the previous function because this allows only
/// certain arcs that fulfil the property that the distance from the start 
/// state to a particular state using any arc would be the same. That is
/// the FST has an inbuilt notion of time.
Lattice* RandPairTimedFst(RandLatticeOptions opts = RandLatticeOptions() ) {
  typedef LatticeArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  Lattice *fst = new Lattice();

 start:

  // Create states.
  vector<StateId> all_states;

  int32 max_time = 1 + (kaldi::Rand() % (opts.n_states-1));
  int32 n_states = 0;

  // Vectors to store the times corresponding to each state and the states
  // at each time
  vector<int32> state_times;
  vector<vector<StateId> > time_states;
  time_states.resize(max_time + 1, vector<StateId>());

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


struct TestForwardBackwardNceOptions {
  bool set_unit_graph_weights;
  bool print_lattice;
  bool check_gradients;
  BaseFloat delta;

  TestForwardBackwardNceOptions() : 
    set_unit_graph_weights(true), 
    print_lattice(true), 
    check_gradients(true),
    delta(1.0e-3) {}

  void Register(OptionsItf *po) {
    po->Register("set-unit-graph-weights", &set_unit_graph_weights,
        "Set graph weights to One()");
    po->Register("print-lattice", &print_lattice,
        "Print Lattice to STDOUT");
    po->Register("check-gradients", &check_gradients,
        "Check gradients by numerical approximation");
    po->Register("delta", &delta,
        "Delta for approximating gradients");
  }
};

CompactLattice *RandDeterminizedCompactLattice(RandLatticeOptions opts) {
  opts.acyclic = true;
  opts.weight_multiplier = 0.25; // impt for the randomly generated weights
  opts.uniq_labels = true;

  while (1) {
    Lattice *fst = RandPairTimedFst(opts);
    CompactLattice *cfst = new CompactLattice;
    if (!DeterminizeLattice(*fst, cfst)) {
      delete fst;
      delete cfst;
      KALDI_WARN << "Determinization failed, trying again.";
    } else {
      delete fst;
      if (cfst->NumStates() != opts.n_states) {
        delete cfst; 
        continue;
      }

      return cfst;
    }
  }
}

void TestForwardBackwardNce(TestForwardBackwardNceOptions opts, 
                            RandLatticeOptions rand_opts) {
  using namespace fst;
  typedef Lattice::Arc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;
  
  TransitionModel tmodel;
  CompactLattice *clat = RandDeterminizedCompactLattice(rand_opts);
  Lattice lat;
  ConvertLattice(*clat, &lat);
  
  bool sorted = fst::TopSort(&lat);
  KALDI_ASSERT(sorted);

  if (opts.print_lattice) {
    KALDI_LOG << "\nComputing Forward Backward on Lattice: ";
  }

  int32 num_arcs = 0;
  std::vector<int32> state_times;
  LatticeStateTimes(lat, &state_times);

  { 
    int32 num_states = lat.NumStates();
    for (StateId s = 0; s < num_states; s++) {
      for (MutableArcIterator<Lattice> aiter(&lat, s); !aiter.Done(); aiter.Next()) {
        Arc arc(aiter.Value());
        if (opts.set_unit_graph_weights) {
          arc.weight.SetValue1(0.0);
        }
        aiter.SetValue(arc);
        if (opts.print_lattice) {
          KALDI_LOG << s << " " << arc.nextstate << " " << arc.ilabel << " " << arc.olabel << " " << arc.weight.Value1() << " + " << arc.weight.Value2();
        }
        num_arcs++;
      }
      Weight f = lat.Final(s);
      if (f != Weight::Zero()) {
        lat.SetFinal(s, Weight::One());
      }
    }
  }
  
  Posterior post;

  Timer timer;
  SignedLogDouble nce_old = LatticeForwardBackwardNce(tmodel, lat, &post);
  KALDI_LOG << "Old code time: " << timer.Elapsed();

  timer.Reset();
  SignedLogDouble nce_new = LatticeForwardBackwardNce(tmodel, lat, &post);
  KALDI_LOG << "New code time: " << timer.Elapsed();

  KALDI_ASSERT(nce_old.ApproxEqual(nce_new));

  while (opts.check_gradients) {
    int32 perturb_arc = RandInt(0, num_arcs);
    int32 perturb_time = -1;
    int32 perturb_tid = -1;
    int32 perturb_state = -1;
    int32 perturb_nextstate = -1;
    double perturb_weight = -1;


    Lattice *lat1 = new Lattice(lat);
    
    int32 num_states = lat.NumStates();

    BaseFloat delta = opts.delta;
    
    bool break_flag = false, continue_flag = false;

    while (delta >= 1e-8) {
      int32 n_arcs = 0;
      for (StateId s = 0; s < num_states; s++) {
        for (MutableArcIterator<Lattice> aiter(lat1, s); !aiter.Done(); aiter.Next(), n_arcs++) {
          if (n_arcs < perturb_arc) continue;
          Arc arc(aiter.Value());
          if (perturb_tid == -1 || arc.ilabel == perturb_tid) {
            double log_p= -arc.weight.Value2();
            arc.weight.SetValue2( -LogAdd(log_p, static_cast<double>(Log(delta))) );
            perturb_weight = -log_p;
            aiter.SetValue(arc);
            perturb_tid = arc.ilabel;
            perturb_time = state_times[s];
            perturb_state = s;
            perturb_nextstate = arc.nextstate;
          }
        }
        if (perturb_tid != -1) { break_flag = true; break; }
      }

      if (perturb_tid == -1) { continue_flag = true; break; }

      Posterior post2;
      SignedLogDouble nce_new = LatticeForwardBackwardNce(tmodel, *lat1, &post2);

      double gradient = 0.0;
      bool found_gradient = false;

      for (int32 i = 0; i < post[perturb_time].size(); i++) {
        if (post[perturb_time][i].first == perturb_tid) {
          gradient += post[perturb_time][i].second;
          found_gradient = true;
        }
      }

      gradient /= Exp(-perturb_weight);

      KALDI_ASSERT(found_gradient);

      double gradient_appx = ((nce_new - nce_old).Value()) / delta;
      KALDI_LOG << "\nPerturbed lattice arc from state " << perturb_state 
        << " to state " << perturb_nextstate << " with tid = " << perturb_tid 
        << "; Computed Gradient is " << gradient << "\n"
        << "Approximated Gradient is (" << nce_new << " - " << nce_old << ") / " << delta << " = " << gradient_appx;

      if (nce_old.LogMagnitude() < -30 || nce_new.LogMagnitude() < -30
          || gradient < 1e-10 || gradient_appx < 1e-10 ) { break_flag = true; break; }

      if(! kaldi::ApproxEqual( gradient_appx, gradient, 0.1 ) ) {
        KALDI_WARN << "There is a large difference in computed and approximated"
          << " gradients; " << gradient << " vs " << gradient_appx << "\n";
      } else break;

      delta /= 10.0;
    }

    if (break_flag) break;
    if (continue_flag) continue;

    if (delta < 1e-8) 
        KALDI_WARN << "There is a large difference in computed and approximated"
          << " gradients; reached delta " << delta;

    break;
  }
}

} // end namespace kaldi

int main(int argc, char** argv) {
  using namespace kaldi;
  using kaldi::int32;
  SetVerboseLevel(4);

  const char *usage = 
        "Test LatticeForwardBackwardNce function\n"
        "Usage: lattice-functions-test [options]\n";
  ParseOptions po(usage);
  
  TestForwardBackwardNceOptions test_opts;
  RandLatticeOptions rand_opts;

  test_opts.Register(&po);
  rand_opts.Register(&po);

  int32 num_iters = 1000;

  po.Register("num-iters", &num_iters,
      "Number of iterations to test");

  po.Read(argc, argv);

  if (po.NumArgs() > 0) {
    po.PrintUsage();
    exit(1);
  }

  for (int32 i = 0; i < num_iters; i++) {
    TestForwardBackwardNce(test_opts, rand_opts);
  }

  KALDI_LOG << "Success.";
}


