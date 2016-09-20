// segmenterbin/segmentation-init-from-lengths.cc

// Copyright 2015   Vimal Manohar (Johns Hopkins University)

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
#include "segmenter/segmenter.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Initialize segmentations from frame lengths file\n"
        "\n"
        "Usage: segmentation-init-from-lengths [options] <lengths-rspecifier> <segmentation-out-wspecifier> \n"
        " e.g.: segmentation-init-from-lengths \"ark:feat-to-len scp:feats.scp ark:- |\" ark:-\n";
    
    int32 label = 1;
    
    ParseOptions po(usage);
 
    po.Register("label", &label, "Assign the segment this class_id");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string lengths_rspecifier = po.GetArg(1),
        segmentation_wspecifier = po.GetArg(2);
    
    SequentialInt32Reader lengths_reader(lengths_rspecifier);
    SegmentationWriter segmentation_writer(segmentation_wspecifier);
    
    int32 num_done = 0;

    for (; !lengths_reader.Done(); lengths_reader.Next()) {
      std::string key = lengths_reader.Key();
      const int32 &len = lengths_reader.Value();
      
      Segmentation seg;
      seg.Emplace(0, len - 1, label);
      
      segmentation_writer.Write(key, seg);
      num_done++;
    }

    KALDI_LOG << "Processed " << num_done << " utterances";

    return (num_done > 0 ? 0 : 1); 
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

