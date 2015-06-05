// nnet2bin/nnet-get-egs-discriminative-unsupervised.cc

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
#include "hmm/transition-model.h"
#include "nnet2/nnet-example-functions.h"
#include "nnet2/am-nnet.h"
#include "lat/lattice-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get unsupervised examples of data for discriminative neural network training;\n"
        "each one corresponds to part of a file, of variable (and configurable\n"
        "length.\n"
        "\n"
        "Usage:  nnet-get-egs-discriminative-unsupervised [options] <model|transition-model> "
        "<features-rspecifier> <lat-rspecifier> "
        "<training-examples-out>\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "nnet-get-egs-discriminative-unsupervised --acoustic-scale=0.1 \\\n"
        "  1.mdl '$feats' 'ark,s,cs:gunzip -c lat.1.gz|' ark:1.degs\n";
    
    std::string ali_rspecifier, oracle_ali_rspecifier, weights_rspecifier;
    bool add_best_path_weights = false;
    BaseFloat acoustic_scale = 1.0, lm_scale = 1.0;

    SplitDiscriminativeExampleConfig split_config;
    
    ParseOptions po(usage);
    split_config.Register(&po);
    
    po.Register("alignment", &ali_rspecifier, "Alignment archive");
    po.Register("oracle", &oracle_ali_rspecifier, "Oracle Alignment archive");
    po.Register("weights", &weights_rspecifier, "Weights archive");
    po.Register("add-best-path-weights", &add_best_path_weights, 
                "Add best path weights to the examples");
    po.Register("acoustic-scale", &acoustic_scale, "Add an acoustic scale "
                " while computing best path");
    po.Register("lm-scale", &lm_scale, "Add an LM scale "
                " while computing best path");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        clat_rspecifier = po.GetArg(3),
        examples_wspecifier = po.GetArg(4);


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
    RandomAccessCompactLatticeReader clat_reader(clat_rspecifier);
    DiscriminativeUnsupervisedNnetExampleWriter example_writer(examples_wspecifier);
    RandomAccessInt32VectorReader ali_reader(ali_rspecifier);
    RandomAccessInt32VectorReader oracle_ali_reader(oracle_ali_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader(weights_rspecifier);

    int32 num_done = 0, num_err = 0;
    int64 examples_count = 0; // used in generating id's.

    SplitExampleStats stats; // diagnostic.

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      if (!clat_reader.HasKey(key)) {
        KALDI_WARN << "No lattice for key " << key;
        num_err++;
        continue;
      }
      CompactLattice clat = clat_reader.Value(key);
      CreateSuperFinal(&clat); // make sure only one state has a final-prob (of One()).
      if (clat.Properties(fst::kTopSorted, true) == 0) {
        TopSort(&clat);
      }

      std::vector<int32> alignment;
      std::vector<int32> oracle_alignment;
      Vector<BaseFloat> weights;

      if (ali_rspecifier != "") {
        if (!ali_reader.HasKey(key)) {
          KALDI_WARN << "No alignment for key " << key;
          num_err++;
          continue;
        }
        alignment = ali_reader.Value(key);
      }
      if (oracle_ali_rspecifier != "") {
        if (!oracle_ali_reader.HasKey(key)) {
          KALDI_WARN << "No oracle alignment for key " << key;
          num_err++;
          continue;
        }
        oracle_alignment = oracle_ali_reader.Value(key);
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
      DiscriminativeUnsupervisedNnetExample eg;

      if (add_best_path_weights) {
        CompactLattice clat_tmp = clat;
        fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat_tmp);
        CompactLattice clat_best_path;
        CompactLatticeShortestPath(clat_tmp, &clat_best_path);
        Lattice best_path;
        ConvertLattice(clat_best_path, &best_path);
        if (best_path.Start() == fst::kNoStateId) {
          KALDI_WARN << "Best-path failed for key " << key;
          continue;
        } else {
          GetLinearSymbolSequence(best_path, &alignment, static_cast<std::vector<int>*>(NULL), static_cast<LatticeWeight*>(NULL));
        }
        Posterior post;

        Lattice lat;
        ConvertLattice(clat, &lat);
        TopSort(&lat);
        LatticeForwardBackward(lat, &post);

        weights.Resize(alignment.size());

        for (int32 i = 0; i < alignment.size(); i++) {
          for(int32 j = 0; j < post[i].size(); j++) {
            if(alignment[i] == post[i][j].first) {
              weights(i) += post[i][j].second;
            }
          }
        }
      }

      if (!LatticeToDiscriminativeUnsupervisedExample(
            alignment, feats,
            clat, weight,
            left_context, right_context, &eg,
            (weights_rspecifier == "" ? NULL : &weights),
            (oracle_ali_rspecifier == "" ? NULL : &oracle_alignment)
            )) {
        KALDI_WARN << "Error converting lattice to example.";
        num_err++;
        continue;
      }
 
      std::vector<DiscriminativeUnsupervisedNnetExample> egs;
      SplitDiscriminativeUnsupervisedExample(split_config, trans_model, 
                                              eg, &egs, &stats);

      KALDI_VLOG(2) << "Split lattice " << key << " into "
                    << egs.size() << " pieces.";
      for (size_t i = 0; i < egs.size(); i++) {
        // Note: excised_egs will be of size 0 or 1.
        std::vector<DiscriminativeUnsupervisedNnetExample> excised_egs;
        ExciseDiscriminativeUnsupervisedExample(split_config, trans_model, egs[i],
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
