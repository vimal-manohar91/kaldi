// latbin/lattice-compute-nce.cc

// Copyright 2015   Vimal Manohar (Johns Hopkins University);

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
#include "hmm/transition-model.h"
#include "hmm/posterior.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Compute NCE for lattices \n"
        "Usage: lattice-compute-nce [options] <trans-model> <lattice-rspecifier> <nce-wspecifier>\n"
        " e.g.: lattice-compute-nce final.mdl ark:1.lats ark,t:1.nce\n";
    
    bool one_silence_class = false;
    BaseFloat lm_scale = 1.0, acoustic_scale = 1.0;
    std::string silence_phones_str, criterion = "nce";
    
    ParseOptions po(usage);
    po.Register("acoustic-scale", &acoustic_scale, "Weighting factor to "
                 "apply to acoustic likelihoods.");
    po.Register("lm-scale", &lm_scale, "Weighting factor to "
                 "apply to lm likelihoods.");
    po.Register("silence-phones", &silence_phones_str,
                 "For MPFE or SMBR, colon-separated list of integer ids of "
                 "silence phones, e.g. 1:2:3");
    po.Register("one-silence-class", &one_silence_class, "If true, newer "
                "behavior which will tend to reduce insertions.");
    po.Register("criterion", &criterion, "Criterion, 'nce'|'empfe'|'esmbr', "
                 "determines the objective function to use.  Should match "
                 "option used when we created the examples.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
  
    std::vector<int32> silence_phones;
    if (silence_phones_str != "") {
      if (!SplitStringToIntegers(silence_phones_str, ":", false,
            &silence_phones)) {
        KALDI_ERR << "Bad value for --silence-phones option: "
                    << silence_phones_str;
      }
    }

    std::string lats_rspecifier = po.GetArg(2),
                nce_wspecifier = po.GetArg(3);
    
    TransitionModel trans;
    ReadKaldiObject(po.GetArg(1), &trans);
    
    int32 n_done = 0;
    
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    BaseFloatWriter nce_writer(nce_wspecifier);

    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      CompactLattice clat = clat_reader.Value();
      clat_reader.FreeCurrent();
      fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat);

      Lattice lat;
      ConvertLattice(clat, &lat);
      TopSort(&lat);

      Posterior post;
      BaseFloat objf;
      if (criterion != "nce") 
        objf = LatticeForwardBackwardEmpeVariants(trans, 
            silence_phones, lat, criterion,
            one_silence_class, &post);
      else 
        objf = LatticeForwardBackwardNce(trans, lat, &post).Value();

      std::vector<int32> state_times;
      int32 num_frames = LatticeStateTimes(lat, &state_times);
      nce_writer.Write(key, objf / num_frames);
      KALDI_LOG << "For " << key << ", average objective function is " 
                << objf / num_frames << " over " << num_frames << " frames";
      n_done++;
    }

    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

