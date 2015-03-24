// latbin/convert-lat.cc

// Copyright 2009-2011  Microsoft Corporation
//                2015  Vimal Manohar (Johns Hopkins University)

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
#include "hmm/hmm-utils.h"
#include "hmm/tree-accu.h"  // for ReadPhoneMap
#include "lat/lattice-functions.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage = 
        "Convert lattices from one decision-tree/model to a subtree\n"
        "Usage:  convert-lat  [options] old-model new-model pdf-map old-lattice-rspecifier new-lattice-wspecifier\n"
        "e.g.: \n"
        " convert-lat old.mdl new.mdl pdf_map ark:old.lat ark:new.lat \n";
        
    std::string phone_map_rxfilename;
    ParseOptions po(usage);
    po.Register("phone-map", &phone_map_rxfilename,
                "File name containing old->new phone mapping (each line is: "
                "old-integer-id new-integer-id)");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string old_model_filename = po.GetArg(1);
    std::string new_model_filename = po.GetArg(2);
    std::string pdf_map_rxfilename = po.GetArg(3);
    std::string old_lat_rspecifier = po.GetArg(4);
    std::string new_lat_wspecifier = po.GetArg(5);

    std::vector<int32> phone_map;
    if (phone_map_rxfilename != "") {  // read phone map.
      ReadPhoneMap(phone_map_rxfilename,
                   &phone_map);
    }

    std::vector<int32> pdf2group;
    {
      bool binary_in;
      Input ki(pdf_map_rxfilename, &binary_in);
      ReadIntegerVector(ki.Stream(), binary_in, &pdf2group);
    }

    SequentialCompactLatticeReader lat_reader(old_lat_rspecifier);
    CompactLatticeWriter lat_writer(new_lat_wspecifier);

    TransitionModel old_trans_model;
    ReadKaldiObject(old_model_filename, &old_trans_model);

    TransitionModel new_trans_model;
    ReadKaldiObject(new_model_filename, &new_trans_model);

    if (!(old_trans_model.GetTopo() == new_trans_model.GetTopo()))
      KALDI_WARN << "Toplogies of models are not equal: "
                 << "conversion may not be correct or may fail.";
    
    int num_success = 0, num_fail = 0;

    for (; !lat_reader.Done(); lat_reader.Next()) {
      std::string key = lat_reader.Key();
      const CompactLattice &clat = lat_reader.Value();
      Lattice lat;
      ConvertLattice(clat, &lat);
      if (ConvertLatticeToNewModel(old_trans_model, 
                                   new_trans_model,
                                   pdf2group, 
                                   (phone_map_rxfilename != "" ? &phone_map : NULL),
                                   &lat)) {
        CompactLattice new_clat;
        ConvertLattice(lat, &new_clat);
        lat_writer.Write(key, new_clat);
        num_success++;
      } else {
        KALDI_WARN << "Could not convert lattice for key " << key;
        num_fail++;
      }
    }

    KALDI_LOG << "Succeeded converting lattices for " << num_success
      <<" files, failed for " << num_fail;

    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

