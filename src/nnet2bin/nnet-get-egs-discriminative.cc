// nnet2bin/nnet-get-egs-discriminative.cc

// Copyright 2012-2013  Johns Hopkins University (author:  Daniel Povey)

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
#include "hmm/transition-model.h"
#include "nnet2/nnet-example-functions.h"
#include "nnet2/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get examples of data for discriminative neural network training;\n"
        "each one corresponds to part of a file, of variable (and configurable)\n"
        "length.\n"
        "\n"
        "Usage:  nnet-get-egs-discriminative [options] <model> "
        "<features-rspecifier> <ali-rspecifier> <den-lat-rspecifier> "
        "<training-examples-out>\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "nnet-get-egs-discriminative --acoustic-scale=0.1 \\\n"
        "  1.mdl '$feats' 'ark,s,cs:gunzip -c ali.1.gz|' 'ark,s,cs:gunzip -c lat.1.gz|' ark:1.degs\n";
    
    SplitDiscriminativeExampleConfig split_config;
    
    std::string oracle_ali_rspecifier, weights_rspecifier, post_rspecifier, num_clat_rspecifier;

    ParseOptions po(usage);
    split_config.Register(&po);
    
    po.Register("oracle", &oracle_ali_rspecifier, "Oracle Alignment archive");
    po.Register("weights", &weights_rspecifier, "Weights archive");
    po.Register("post", &post_rspecifier, "Numerator posteriors");
    po.Register("num-clat", &num_clat_rspecifier, "Numerator compact lattice");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        ali_rspecifier = po.GetArg(3),
        clat_rspecifier = po.GetArg(4),
        examples_wspecifier = po.GetArg(5);


    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }

    int32 left_context = am_nnet.GetNnet().LeftContext(),
        right_context = am_nnet.GetNnet().RightContext();

    
    // Read in all the training files.
    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    RandomAccessInt32VectorReader ali_reader(ali_rspecifier);
    RandomAccessCompactLatticeReader clat_reader(clat_rspecifier);
    DiscriminativeNnetExampleWriter example_writer(examples_wspecifier);
    RandomAccessInt32VectorReader oracle_ali_reader(oracle_ali_rspecifier);
    RandomAccessPosteriorReader posterior_reader(post_rspecifier);
    RandomAccessCompactLatticeReader num_clat_reader(num_clat_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader(weights_rspecifier);

    int32 num_done = 0, num_err = 0;
    int64 examples_count = 0; // used in generating id's.
    
    SplitExampleStats stats; // diagnostic.
    
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      if (!ali_reader.HasKey(key)) {
        KALDI_WARN << "No pdf-level posterior for key " << key;
        num_err++;
        continue;
      }
      const std::vector<int32> &alignment = ali_reader.Value(key);
      if (!clat_reader.HasKey(key)) {
        KALDI_WARN << "No denominator lattice for key " << key;
        num_err++;
        continue;
      }
      CompactLattice clat = clat_reader.Value(key);
      CreateSuperFinal(&clat); // make sure only one state has a final-prob (of One()).
      if (clat.Properties(fst::kTopSorted, true) == 0) {
        TopSort(&clat);
      }

      std::vector<int32> oracle_alignment;
      Vector<BaseFloat> weights;
      Posterior post;
      CompactLattice num_clat;

      if (oracle_ali_rspecifier != "") {
        if (!oracle_ali_reader.HasKey(key)) {
          KALDI_WARN << "No oracle alignment for key " << key;
          num_err++;
          continue;
        }
        oracle_alignment = oracle_ali_reader.Value(key);
      }

      if (post_rspecifier != "") {
        if (!posterior_reader.HasKey(key)) {
          KALDI_WARN << "No posterior for key " << key;
          num_err++;
          continue;
        }
        post = posterior_reader.Value(key);
      }

      if (num_clat_rspecifier != "") {
        if (!num_clat_reader.HasKey(key)) {
          KALDI_WARN << "No numerator lattice for key " << key;
          num_err++;
          continue;
        }
        num_clat = num_clat_reader.Value(key);
        CreateSuperFinal(&num_clat); // make sure only one state has a final-prob (of One()).
        if (num_clat.Properties(fst::kTopSorted, true) == 0) {
          TopSort(&num_clat);
        }
      }
      
      if (weights_rspecifier != "") {
        if (!weights_reader.HasKey(key)) { 
          KALDI_WARN << "No weights for key " << key;
          num_err++;
          continue;
        }
        weights = weights_reader.Value(key);
      }

      BaseFloat weight = 1.0;
      DiscriminativeNnetExample eg;

      Vector<BaseFloat> *weights_ptr = (weights_rspecifier != "" ? &weights : NULL);
      std::vector<int32> *oracle_ptr = (oracle_ali_rspecifier != "" ? &oracle_alignment : NULL);

      if (num_clat_rspecifier != "" &&
          !LatticeToDiscriminativeExample(alignment, num_clat, feats, clat, weight,
            left_context, right_context, &eg, weights_ptr, oracle_ptr)) {
        KALDI_WARN << "Error converting lattice to example.";
        num_err++;
        continue;
      } else if (post_rspecifier != "" &&
          !LatticeToDiscriminativeExample(alignment, post, feats, clat, weight,
            left_context, right_context, &eg, weights_ptr, oracle_ptr)) {
        KALDI_WARN << "Error converting lattice to example.";
        num_err++;
        continue;
      } else if (!LatticeToDiscriminativeExample(alignment, feats, clat, weight,
            left_context, right_context, &eg, weights_ptr, oracle_ptr)) {
        KALDI_WARN << "Error converting lattice to example.";
        num_err++;
        continue;
      }
      
      std::vector<DiscriminativeNnetExample> egs;
      SplitDiscriminativeExample(split_config, trans_model, eg,
                                 &egs, &stats);
      
      KALDI_VLOG(2) << "Split lattice " << key << " into "
                    << egs.size() << " pieces.";
      for (size_t i = 0; i < egs.size(); i++) {
        // Note: excised_egs will be of size 0 or 1.
        std::vector<DiscriminativeNnetExample> excised_egs;
        ExciseDiscriminativeExample(split_config, trans_model, egs[i],
                                    &excised_egs, &stats);
        for (size_t j = 0; j < excised_egs.size(); j++) {
          std::ostringstream os;
          os << (examples_count++);
          std::string example_key = os.str();
          example_writer.Write(example_key, excised_egs[j]);
        }
      }
      num_done++;
    }

    if (num_done > 0) stats.Print();
    
    KALDI_LOG << "Finished generating examples, "
              << "successfully processed " << num_done
              << " feature files, " << num_err << " had errors.";
    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
