// segmenterbin/segmentation-merge-recordings.cc

// Copyright 2016   Vimal Manohar (Johns Hopkins University)

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
        "Merge segmentations of different recordings into one segmentation "
        "using new2oldlist_map\n"
        "\n"
        "Usage: segmentation-merge-recordings [options] <segmentation-rspecifier> <new2old-list-map> <segmentation-wspecifier>\n"
        " e.g.: segmentation-merge-recordings ark:in_seg.ark ark:new2old_map ark:out_seg.ark\n";
    
    ParseOptions po(usage);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
  
    std::string segmentation_rspecifier = po.GetArg(1),
                new2old_list_rspecifier = po.GetArg(2),
                segmentation_wspecifier = po.GetArg(3);

    SequentialTokenVectorReader new2old_reader(new2old_list_rspecifier);
    RandomAccessSegmentationReader  segmentation_reader(segmentation_rspecifier);
    SegmentationWriter segmentation_writer(segmentation_wspecifier);
    
    int32 num_new_segmentations = 0, num_old_segmentations = 0;
    int64 num_segments = 0, num_err = 0;

    for (; !new2old_reader.Done(); new2old_reader.Next()) {
      const std::vector<std::string> &old_key_list = new2old_reader.Value();
      const std::string &new_key = new2old_reader.Key();
 
      KALDI_ASSERT(old_key_list.size() > 0);

      Segmentation seg;

      for (std::vector<std::string>::const_iterator it = old_key_list.begin();
            it != old_key_list.end(); ++it) {
        num_old_segmentations++;
        
        if (!segmentation_reader.HasKey(*it)) {
          KALDI_WARN << "Could not find key " << *it << " in " 
                     << "old segmentation " << segmentation_rspecifier;
          num_err++;
          continue;
        }
        
        const Segmentation &this_seg = segmentation_reader.Value(*it);

        num_segments += seg.InsertFromSegmentation(this_seg, 0, NULL);
      }
      seg.Sort();
      segmentation_writer.Write(reco_id, seg);
      
      num_new_segmentations++;
    }

    KALDI_LOG << "Merged " << num_old_segmentations << " old segmentations "
              << "into " << num_new_segmentations << " new segmentations; "
              << "created overall " << num_segments << " segments; "
              << "failed to merge " << num_err << " old segmentations";

    return (num_new_segmentations > 0 && num_err < num_old_segmentations / 2);

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}






