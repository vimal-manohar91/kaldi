// chainbin/chain-lattice-to-post.cc

// Copyright      2017  Vimal Manohar

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

#include <sstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include "lat/lattice-functions.h"
#include "chain/chain-supervision.h"
#include "chain/chain-supervision-splitter.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::chain;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Do forward-backward and collect pdf posteriors over lattices.\n"
        "The labels are converted to a 1-index i.e. pdf-id + 1\n"
        "An FST with labels as the 1-indexed pdf-ids can be optionally "
        "provided to interpolate with the LM scores from lattice.\n"
        "Usage:  chain-lattice-to-post [options] [<fst-in>] <trans-model> <lattice-rspecifier> "
        "<post-wspecifier>\n"
        "\n";

    BaseFloat acoustic_scale = 1.0, fst_scale = 0.0;

    ParseOptions po(usage);
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("fst-scale", &fst_scale,
                "Scaling factor for the <fst-in> that will interpolated "
                "with the lattice."
                "Effectively this is (1-fst_scale) * lattice-graph-cost + fst_scale * fst-costs");
    po.Read(argc, argv);

    if (po.NumArgs() != 3 && po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string trans_model_rxfilename,
      lattice_rspecifier,
      fst_rxfilename,
      post_wspecifier;

    if (po.NumArgs() == 3) {
      trans_model_rxfilename = po.GetArg(1);
      lattice_rspecifier = po.GetArg(2);
      post_wspecifier = po.GetArg(3);
    } else {
      fst_rxfilename = po.GetArg(1);
      trans_model_rxfilename = po.GetArg(2);
      lattice_rspecifier = po.GetArg(3);
      post_wspecifier = po.GetArg(4);
    }

    TransitionModel trans_model;
    ReadKaldiObject(trans_model_rxfilename, &trans_model);

    fst::StdVectorFst fst;
    if (!fst_rxfilename.empty()) {
      ReadFstKaldi(fst_rxfilename, &fst);
      KALDI_ASSERT(fst.NumStates() > 0);

      if (fst_scale < 0.0 || fst_scale > 1.0) {
        KALDI_ERR << "Invalid fst-scale; must be in [0.0, 1.0)";
      }

      if (fst_scale != 1.0) {
        fst::ApplyProbabilityScale(fst_scale, &fst);
      }
    }

    fst::RmEpsilon(&fst);
    fst::ArcSort(&fst, fst::ILabelCompare<fst::StdArc>());

    SequentialLatticeReader lattice_reader(lattice_rspecifier);
    PosteriorWriter posterior_writer(post_wspecifier);

    int32 num_done = 0, num_fail = 0;
    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();

      Lattice lat = lattice_reader.Value();

      fst::ScaleLattice(fst::LatticeScale(1.0 - fst_scale, acoustic_scale), &lat);

      Posterior graph_post;
      bool status = LatticeToNumeratorPost(lat, trans_model, fst,
                                           &graph_post, key);
      if (!status) {
        num_fail++;
        continue;
      }

      posterior_writer.Write(key, graph_post);
      num_done++;
    }

    KALDI_LOG << "Converted " << num_done << " lattices to posteriors; "
              << "failed for " << num_fail;

    return num_done > 0 ? 0 : 1;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
