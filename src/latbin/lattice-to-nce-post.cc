// latbin/lattice-to-nce-post.cc

// Copyright 2014   Vimal Manohar

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "base/kaldi-types-extra.h"

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    typedef kaldi::SignedLogReal<double> SignedLogDouble;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;
    using namespace kaldi;
  
    const char *usage =
        "Do forward-backward and collect frame level NCE posteriors over\n" 
        "lattices, which can be fed into gmm-acc-stats to do \n"
        "unsupervised discriminative training.\n"
        "Usage: lattice-to-nce-post [options] <model> \n"
        " <lats-rspecifier> <posteriors-wspecifier> \n"
        "e.g.: lattice-to-nce-post --acoustic-scale=0.1 1.mdl \n"
        " ark:1.lats ark:1.post\n";

    kaldi::BaseFloat acoustic_scale = 1.0, lm_scale = 1.0;
    //std::string silence_phones_str;
    kaldi::ParseOptions po(usage);
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("lm-scale", &lm_scale,
                "Scaling factor for \"graph costs\" (including LM costs)");
    //po.Register("silence-phones", &silence_phones_str,
    //            "Colon-separated list of integer id's of silence phones, e.g. 46:47");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
 
    /*
    std::vector<int32> silence_phones;
    if (!kaldi::SplitStringToIntegers(silence_phones_str, ":", false, &silence_phones))
      KALDI_ERR << "Invalid silence-phones string " << silence_phones_str;
    kaldi::SortAndUniq(&silence_phones);
    if (silence_phones.empty())
      KALDI_WARN << "No silence phones specified, make sure this is what you intended.";
    */

    if (acoustic_scale == 0.0)
      KALDI_ERR << "Do not use a zero acoustic scale (cannot be inverted)";

    std::string model_filename = po.GetArg(1), 
        lats_rspecifier = po.GetArg(2),
        posteriors_wspecifier = po.GetArg(3);

    SequentialLatticeReader lattice_reader(lats_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);
    
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
    }

    int32 num_done = 0;
    double total_lat_nce = 0.0, lat_nce;
    double total_time = 0, lat_time;


    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      kaldi::Lattice lat = lattice_reader.Value();
      lattice_reader.FreeCurrent();
      if (acoustic_scale != 1.0 || lm_scale != 1.0)
        fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &lat);
      
      kaldi::uint64 props = lat.Properties(fst::kFstProperties, false);
      if (!(props & fst::kTopSorted)) {
        if (fst::TopSort(&lat) == false)
          KALDI_ERR << "Cycles detected in lattice.";
      }
      
      Posterior post;
      // We don't need to scale posteriors because it the same constant
      // of acoustic scale for the other posteriors like MPE numerator and
      // denominator posteriors.
      lat_nce = LatticeForwardBackwardNCE(trans_model, lat, &post).Value();
      total_lat_nce += lat_nce;
      lat_time = post.size();
      total_time += lat_time;
      KALDI_VLOG(2) << "Processed lattice for utterance: " << key << "; found "
                    << lat.NumStates() << " states and " << fst::NumArcs(lat)
                    << " arcs. Negative Conditional Entropy = " 
                    << (lat_nce/lat_time)
                    << " over " << lat_time << " frames.";
      posterior_writer.Write(key, post);
      num_done++; 
    }

    KALDI_LOG << "Overall average Negative Conditional Entropy is "
              << (total_lat_nce/total_time) << " over " << total_time
              << " frames.";
    KALDI_LOG << "Done " << num_done << " lattices.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

