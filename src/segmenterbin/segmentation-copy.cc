// segmenterbin/segmentation-copy.cc

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
        "Copy segmentation or archives of segmentation\n"
        "\n"
        "Usage: segmentation-copy [options] (segmentation-in-rspecifier|segmentation-in-rxfilename) (segmentation-out-wspecifier|segmentation-out-wxfilename)\n"
        " e.g.: segmentation-copy --binary=false foo -\n"
        "   segmentation-copy ark:1.ali ark,t:-\n";
    
    bool binary = true;
    std::string label_map_rxfilename;

    ParseOptions po(usage);
    
    po.Register("binary", &binary, 
                "Write in binary mode (only relevant if output is a wxfilename)");
    po.Register("label-map", &label_map_rxfilename,
                "File with mapping from old to new labels");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string segmentation_in_fn = po.GetArg(1),
                segmentation_out_fn = po.GetArg(2);

    unordered_map<int32, int32> label_map;
    if (!label_map_rxfilename.empty()) {
      Input ki(label_map_rxfilename);
      std::string line;
      while (std::getline(ki.Stream(), line)) {
        std::vector<std::string> splits;
        SplitStringToVector(line, " ", true, &splits);

        if (splits.size() != 2) 
          KALDI_ERR << "Invalid format of line " << line 
                    << " in " << label_map_rxfilename;

        label_map[std::atoi(splits[0].c_str())] = std::atoi(splits[1].c_str());
      }
    }

    // all these "fn"'s are either rspecifiers or filenames.

    bool in_is_rspecifier =
        (ClassifyRspecifier(segmentation_in_fn, NULL, NULL)
         != kNoRspecifier),
        out_is_wspecifier =
        (ClassifyWspecifier(segmentation_out_fn, NULL, NULL, NULL)
         != kNoWspecifier);

    if (in_is_rspecifier != out_is_wspecifier)
      KALDI_ERR << "Cannot mix regular files and archives";
    
    int64  num_done = 0, num_err = 0;
    
    if (!in_is_rspecifier) {
      Segmentation seg;
      {
        bool binary_in;
        Input ki(segmentation_in_fn, &binary_in);
        seg.Read(ki.Stream(), binary_in);
      }

      if (!label_map_rxfilename.empty())
        seg.RelabelSegmentsUsingMap(label_map);

      Output ko(segmentation_out_fn, binary);
      seg.Write(ko.Stream(), binary);
      KALDI_LOG << "Copied segmentation to " << segmentation_out_fn;
      return 0;
    } else {
      SegmentationWriter writer(segmentation_out_fn); 
      SequentialSegmentationReader reader(segmentation_in_fn);
      for (; !reader.Done(); reader.Next(), num_done++) {
        if (label_map_rxfilename.empty())
          writer.Write(reader.Key(), reader.Value());
        else {
          Segmentation seg = reader.Value();
          seg.RelabelSegmentsUsingMap(label_map);
          writer.Write(reader.Key(), seg);
        }
      }

      KALDI_LOG << "Copied " << num_done << " segmentation; failed with "
                << num_err << " segmentations";
      return (num_done != 0 ? 0 : 1);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

