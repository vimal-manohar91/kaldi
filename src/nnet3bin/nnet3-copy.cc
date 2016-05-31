// nnet3bin/nnet3-copy.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)
//           2015  Xingyu Na

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

#include <typeinfo>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-simple-component.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy 'raw' nnet3 neural network to standard output\n"
        "Also supports setting all the learning rates to a value\n"
        "(the --learning-rate option)\n"
        "\n"
        "Usage:  nnet3-copy [options] <nnet-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet3-copy --binary=false 0.raw text.raw\n";

    bool binary_write = true;
    BaseFloat learning_rate = -1;
    std::string learning_rates_csl;
    std::string learning_rate_factors_csl;
    BaseFloat scale = 1.0;
    std::string rename_nodes_wxfilename;
    std::string add_prefix_to_names;
    bool convert_affine_to_fixed_affine = false;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("learning-rate", &learning_rate,
                "If supplied, all the learning rates of updatable components"
                "are set to this value.");
    po.Register("learning-rates", &learning_rates_csl,
                "If supplied, set the actual learning rates of the "
                "updatable components to these set of values");
    po.Register("learning-rate-factors", &learning_rate_factors_csl,
                "If supplied, set the learning rate factors of the "
                "updatable components to these set of values");
    po.Register("scale", &scale, "The parameter matrices are scaled"
                " by the specified value.");
    po.Register("rename-nodes-wxfilename", &rename_nodes_wxfilename, 
                "A mapping from old node name to new name");
    po.Register("add-prefix-to-names", &add_prefix_to_names,
                "Add prefix to all the node names");
    po.Register("convert-affine-to-fixed-affine", &convert_affine_to_fixed_affine,
                "Convert components with zero learning rate factor into "
                "fixed affine component");

    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string raw_nnet_rxfilename = po.GetArg(1),
                raw_nnet_wxfilename = po.GetArg(2);
    
    Nnet nnet;
    ReadKaldiObject(raw_nnet_rxfilename, &nnet);
    
    if (learning_rate >= 0)
      SetLearningRate(learning_rate, &nnet);
    
    if (scale != 1.0)
      ScaleNnet(scale, &nnet);

    if (!learning_rate_factors_csl.empty()) {
      std::vector<BaseFloat> learning_rate_factors;
      SplitStringToFloats(learning_rate_factors_csl, ":,", true, &learning_rate_factors);

      Vector<BaseFloat> temp(learning_rate_factors.size());
      for (size_t i = 0; i < learning_rate_factors.size(); i++) {
        temp(i) = learning_rate_factors[i];
      }
      SetLearningRateFactors(temp, &nnet);
    }
    
    if (!learning_rates_csl.empty()) {
      std::vector<BaseFloat> learning_rates;
      SplitStringToFloats(learning_rates_csl, ":,", true, &learning_rates);

      Vector<BaseFloat> temp(learning_rates.size());
      for (size_t i = 0; i < learning_rates.size(); i++) {
        temp(i) = learning_rates[i];
      }
      SetLearningRates(temp, &nnet);
    }

    if (!rename_nodes_wxfilename.empty()) {
      unordered_map<std::string, std::string, StringHasher> node_names_map;

      Input ki(rename_nodes_wxfilename);
      std::string line;

      while (std::getline(ki.Stream(), line)) {
        std::vector<std::string> split_line;
        SplitStringToVector(line, " \t\r", true, &split_line);

        if (split_line.size() == 0)
          continue;

        if (split_line.size() != 2)
          KALDI_ERR << "Invalid line in " << rename_nodes_wxfilename;

        node_names_map[split_line[0]] = split_line[1]; 
      }

      nnet.RenameNodes(node_names_map);
    }

    if (!add_prefix_to_names.empty()) {
      nnet.AddPrefixToNames(add_prefix_to_names);
    }

    if (convert_affine_to_fixed_affine) {
      ConvertAffineToFixedAffine(&nnet, true);
    }

    WriteKaldiObject(nnet, raw_nnet_wxfilename, binary_write);
    KALDI_LOG << "Copied raw neural net from " << raw_nnet_rxfilename
              << " to " << raw_nnet_wxfilename;

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
