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
#include "fstext/rand-fst.h"
#include "hmm/transition-model.h"
#include "lat/determinize-lattice-pruned.h"
#include "fstext/lattice-utils.h"
#include "fstext/fst-test-utils.h"
#include "lat/lattice-functions.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "lat/lattice-functions.h"
#include "hmm/transition-model.h"
#include "util/stl-utils.h"
#include "base/kaldi-math.h"

namespace kaldi {
using namespace fst;

struct TestForwardBackwardNCEOptions {
  bool set_unit_graph_weights;
  bool print_lattice;
  bool check_gradients;
  BaseFloat delta;

  TestForwardBackwardNCEOptions() : 
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

CompactLattice *RandDeterminizedCompactLattice() {
  RandFstOptions opts;
  opts.acyclic = true;
  opts.n_states = 4;
  opts.n_final = 1;
  opts.n_arcs = 4;
  opts.weight_multiplier = 0.5; // impt for the randomly generated weights

  while (1) {
    Lattice *fst = fst::RandPairFst<LatticeArc>(opts);
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

void TestForwardBackwardNCE(TestForwardBackwardNCEOptions opts) {
  using namespace fst;
  typedef Lattice::Arc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;
  
  TransitionModel tmodel;
  CompactLattice *clat = RandDeterminizedCompactLattice();
  Lattice lat;
  ConvertLattice(*clat, &lat);
  
  bool sorted = fst::TopSort(&lat);
  KALDI_ASSERT(sorted);

  if (opts.print_lattice) {
    KALDI_LOG << "Computing Forward Backward on Lattice: ";
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
          aiter.SetValue(arc);
        }
        if (opts.print_lattice) {
          KALDI_LOG << s << " " << arc.nextstate << " " << arc.weight.Value1() << " + " << arc.weight.Value2();
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
  SignedLogDouble nce_old = LatticeForwardBackwardNCE(tmodel, lat, &post);

  while (opts.check_gradients) {
    int32 perturb_arc = RandInt(0, num_arcs);
    int32 perturb_time = -1;
    int32 perturb_tid = -1;
    int32 perturb_state = -1;
    int32 perturb_nextstate = -1;
    double perturb_weight = -1;


    Lattice *lat1 = new Lattice(lat);
    
    int32 num_states = lat.NumStates();

    int32 n_arcs = 0;
    for (StateId s = 0; s < num_states; s++) {
      for (MutableArcIterator<Lattice> aiter(lat1, s); !aiter.Done(); aiter.Next(), n_arcs++) {
        if (n_arcs < perturb_arc) continue;
        Arc arc(aiter.Value());
        if (arc.ilabel == 0) continue;
        if (perturb_tid == -1 || arc.ilabel == perturb_tid) {
          double log_p= -arc.weight.Value2();
          arc.weight.SetValue2( -LogAdd(log_p, static_cast<double>(Log(opts.delta))) );
          perturb_weight = -log_p;
          aiter.SetValue(arc);
          perturb_tid = arc.ilabel;
          perturb_time = state_times[s];
          perturb_state = s;
          perturb_nextstate = arc.nextstate;
        }
      }
      if (perturb_tid != -1) break;
    }

    if (perturb_tid == -1) continue;

    Posterior post2;
    SignedLogDouble nce_new = LatticeForwardBackwardNCE(tmodel, *lat1, &post2);

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

    double gradient_appx = ((nce_new - nce_old).Value()) / opts.delta;
    KALDI_LOG << "Perturbed lattice arc from " << perturb_state << " to " 
      << perturb_nextstate << " with tid = " << perturb_tid << "; "
      << "Computed Gradient is " << gradient << "\n"
      << "Actual Gradient is (" << nce_new << " - " << nce_old << ") / " << opts.delta << " = " << gradient_appx << "\n";

    if (nce_old.LogMagnitude() < -30 || nce_new.LogMagnitude() < -30) break;

    KALDI_ASSERT( kaldi::ApproxEqual( gradient_appx, gradient, 0.1 ) ); 

    break;
  }
}

} // end namespace kaldi

int main(int argc, char** argv) {
  using namespace kaldi;
  using kaldi::int32;
  SetVerboseLevel(4);

  const char *usage = 
        "Test LatticeForwardBackwardNCE function\n"
        "Usage: lattice-functions-test [options]\n";
  ParseOptions po(usage);
  
  TestForwardBackwardNCEOptions opts;
  opts.Register(&po);

  po.Read(argc, argv);

  if (po.NumArgs() > 0) {
    po.PrintUsage();
    exit(1);
  }

  for (int32 i = 0; i < 1000; i++) {
    TestForwardBackwardNCE(opts);
  }

  KALDI_LOG << "Success.";
}

