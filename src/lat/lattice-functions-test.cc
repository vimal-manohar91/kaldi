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

void TestForwardBackwardNCE(bool set_unit_graph_weights = true, bool print_lattice = true) {
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

  if (print_lattice) {
    KALDI_LOG << "Computing Forward Backward on Lattice: ";
  }
  if (set_unit_graph_weights || print_lattice) {
    int32 num_states = lat.NumStates();
    for (StateId s = 0; s < num_states; s++) {
      for (MutableArcIterator<Lattice> aiter(&lat, s); !aiter.Done(); aiter.Next()) {
        Arc arc(aiter.Value());
        if (set_unit_graph_weights) {
          arc.weight.SetValue1(0.0);
          aiter.SetValue(arc);
        }
        if (print_lattice) {
          KALDI_LOG << s << " " << arc.nextstate << " " << arc.weight.Value1() << " + " << arc.weight.Value2();
        }
      }
      Weight f = lat.Final(s);
      if (f != Weight::Zero()) {
        lat.SetFinal(s, Weight::One());
      }
    }
  }

  Posterior post;
  LatticeForwardBackwardNCE(tmodel, lat, &post);
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

  bool set_unit_graph_weights = true;

  po.Register("set-unit-graph-weights", &set_unit_graph_weights,
              "Set graph weights to One()");
  po.Read(argc, argv);

  if (po.NumArgs() > 0) {
    po.PrintUsage();
    exit(1);
  }

  for (int32 i = 0; i < 1000; i++) {
    TestForwardBackwardNCE(set_unit_graph_weights);
  }

  KALDI_LOG << "Success.";
}

