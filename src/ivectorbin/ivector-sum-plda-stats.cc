// ivectorbin/ivector-sum-acc-plda-stats.cc

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
        "Add plda stats from multiple sources. \n"
        "\n"
        "Usage:  ivector-acc-plda-stats [options] <plda-stats-out> <plda-stats-in1> <plda-stats-in2> ... \n"
        "e.g.: \n"
        " ivector-acc-plda-stats - plda_stats1 plda_stats2\n";

    ParseOptions po(usage);

    bool binary = true;
    bool add_offset_scatter = true;

    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("add-offset-scatter", &add_offset_scatter,
                "Set this to false to sum only the between-class stats "
                "and keep the within-class stats only from the "
                "first accumulator.");

    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda_stats_wxfilename = po.GetArg(1);

    PldaStats plda_stats;

    {
      std::vector<std::string> parts;
      SplitStringToVector(po.GetArg(2), ":", false, &parts);
      KALDI_ASSERT(parts.size() == 1 || parts.size() == 2);
      ReadKaldiObject(parts[0], &plda_stats);
      BaseFloat this_weight = 1.0;
      if (parts.size() == 2) 
        KALDI_ASSERT(ConvertStringToReal(parts[1], &this_weight));
      plda_stats.Scale(this_weight);
    }

    for (int32 i = 3; i <= po.NumArgs(); i++) {
      PldaStats this_plda_stats;
      std::vector<std::string> parts;
      SplitStringToVector(po.GetArg(i), ":", false, &parts);
      KALDI_ASSERT(parts.size() == 1 || parts.size() == 2);
      BaseFloat this_weight = 1.0;
      if (parts.size() == 2) 
        KALDI_ASSERT(ConvertStringToReal(parts[2], &this_weight));
      ReadKaldiObject(parts[0], &this_plda_stats);
      plda_stats.Sum(this_plda_stats, this_weight, add_offset_scatter);
    }

    plda_stats.Sort();

    WriteKaldiObject(plda_stats, plda_stats_wxfilename, binary);
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


