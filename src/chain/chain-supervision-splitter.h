// chain/chain-supervision-splitter.h

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

#ifndef KALDI_CHAIN_CHAIN_SUPERVISION_SPILTTER_H_
#define KALDI_CHAIN_CHAIN_SUPERVISION_SPILTTER_H_

#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"
#include "chain/chain-supervision.h"

namespace kaldi {
namespace chain {

typedef fst::ArcTpl<LatticeWeight> LatticeArc;
typedef fst::VectorFst<LatticeArc> Lattice;

struct SupervisionLatticeSplitterOptions {
  BaseFloat acoustic_scale;
  bool normalize;
  bool convert_to_unconstrained;
  bool debug;
  BaseFloat extra_scale;
  bool only_scale_graph;

  SupervisionLatticeSplitterOptions(): 
    acoustic_scale(1.0), normalize(true),
    convert_to_unconstrained(false), debug(false), 
    extra_scale(0.0), only_scale_graph(true) { }

  void Register(OptionsItf *opts) {
    opts->Register("acoustic-scale", &acoustic_scale,
                   "Apply acoustic scale on the lattices before splitting.");
    opts->Register("normalize", &normalize,
                   "Normalize the initial and final scores added to split "
                   "lattices");
    opts->Register("convert-to-unconstrained", &convert_to_unconstrained,
                   "If this is true, then self-loop transitions in the "
                   "supervision are replaced by self-loops");
    opts->Register("debug", &debug,
                   "Run some debug test codes");
    opts->Register("extra-scale", &extra_scale,
                   "This is an extra scale that is added to the "
                   "costs (including acoustic costs) from the lattice before "
                   "adding them to the supervision. "
                   "The default is 0, which means no acoustic cost and only "
                   "the raw graph cost is included from the lattice "
                   "in the supervision.");
    opts->Register("only-scale-graph", &only_scale_graph,
                   "Only scale the graph and not the acoustic score. "
                   "This is more appropriate. "
                   "Also sets normalize to false.");
  }
};

class SupervisionLatticeSplitter {
 public:
  SupervisionLatticeSplitter(const SupervisionLatticeSplitterOptions &opts,
                             const SupervisionOptions &sup_opts,
                             const TransitionModel &trans_model);

  bool LoadLattice(const Lattice &lat);

  bool GetFrameRangeSupervision(int32 begin_frame, int32 frames_per_sequence,
                                chain::Supervision *supervision,
                                Lattice *lat = NULL,
                                Lattice *raw_range_lat = NULL) const;

  int32 NumFrames() const { return lat_scores_.num_frames; }

  // A structure used to store the forward and backward scores
  // and state times of a lattice
  struct LatticeInfo {
    // These values are stored in log.
    std::vector<double> alpha;
    std::vector<double> beta;
    std::vector<int32> state_times;
    std::vector<std::vector<std::pair<int32, BaseFloat> > > post;
    int32 num_frames;

    void Reset() {
      alpha.clear(); 
      beta.clear(); 
      state_times.clear(); 
      post.clear();
    }

    void Check() const;
  };

  const Lattice& GetLattice() const { return lat_; }

  const fst::StdVectorFst& ToleranceFst() const { return tolerance_fst_; }
 private:
  // Creates an output lattice covering frames begin_frame <= t < end_frame,
  // assuming that the corresponding state-range that we need to
  // include, begin_state <= s < end_state has been included.
  // (note: the output lattice will also have two special initial and final
  // states).
  void CreateRangeLattice(int32 begin_frame, int32 end_frame,
                          Lattice *out_lat) const;

  void PostProcessLattice(Lattice *out_lat) const;

  bool GetSupervision(const Lattice &out_lat, Supervision *supervision) const;

  // Function to compute lattice scores for a lattice
  void ComputeLatticeScores();
  
  // Prepare lattice :
  // 1) Order states in breadth-first search order
  // 2) Compute states times, which must be a strictly non-decreasing vector
  // 3) Compute lattice alpha and beta scores
  bool PrepareLattice();
  
  const SupervisionOptions &sup_opts_;
  
  const SupervisionLatticeSplitterOptions &opts_;

  const TransitionModel &trans_model_;

  fst::StdVectorFst tolerance_fst_;
  void MakeToleranceEnforcerFst();

  // Copy of the lattice loaded using LoadLattice(). 
  // This is required because the lattice states
  // need to be ordered in breadth-first search order.
  Lattice lat_;

  // LatticeInfo object for lattice.
  // This will be computed when PrepareLattice function is called.
  LatticeInfo lat_scores_;
};

void GetToleranceEnforcerFst(const SupervisionOptions &opts, const TransitionModel &trans_model, fst::StdVectorFst *tolerance_fst);

}
}

#endif  // KALDI_CHAIN_CHAIN_SUPERVISION_SPLITTER_H_
