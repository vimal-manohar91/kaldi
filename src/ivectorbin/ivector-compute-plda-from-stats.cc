// ivectorbin/ivector-compute-plda-from-stats.cc

// Copyright 2013  Daniel Povey
//           2017  Vimal Manohar

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
#include "ivector/plda.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Computes a Plda object (for Probabilistic Linear Discriminant Analysis)\n"
        "from accumulated Plda stats.\n"
        "\n"
        "Usage:  ivector-compute-plda-from-stats [options] <plda-stats-in> "
        "<plda-out>\n"
        "e.g.: \n"
        " ivector-compute-plda-from-stats plda_stats plda\n"
        "See also: ivector-compute-plda, ivector-acc-plda-stats\n";

    ParseOptions po(usage);

    bool binary = true;

    PldaEstimationConfig plda_config;
    plda_config.Register(&po);

    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda_stats_rxfilename = po.GetArg(1),
        plda_wxfilename = po.GetArg(2);

    PldaStats plda_stats;
    ReadKaldiObject(plda_stats_rxfilename, &plda_stats);
    plda_stats.Sort();

    Plda plda;
    PldaEstimator plda_estimator(plda_stats);
    plda_estimator.Estimate(plda_config, &plda);
    WriteKaldiObject(plda, plda_wxfilename, binary);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

