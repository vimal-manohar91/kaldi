// segmenterbin/segmentation-create-overlapped-subsegments.cc

// Copyright 2015-16   Vimal Manohar (Johns Hopkins University)

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
#include "segmenter/segmentation-utils.h"

namespace kaldi {
namespace segmenter {

void CreateOverlappedSubSegments(
    const Segmentation &primary_segmentation,
    const Segmentation &secondary_segmentation, int32 secondary_label,
    Segmentation *out_segmentation) {
  KALDI_ASSERT(secondary_segmentation.Dim() > 0);

#ifdef KALDI_PARANOID
  KALDI_ASSERT(IsNonOverlapping(secondary_segmentation));
#endif
  
  std::vector<int32> alignment;
  ConvertToAlignment(secondary_segmentation, -1,
                     -1, 0, &alignment);
   
  for (SegmentList::const_iterator it = primary_segmentation.Begin();
        it != primary_segmentation.End(); ++it) {

    if (it->end_frame >= alignment.size()) {
      alignment.resize(it->end_frame + 1, -1);
    }

    Segmentation filter_segmentation;
    InsertFromAlignment(alignment, it->start_frame, it->end_frame + 1,
                        0, &filter_segmentation, NULL);

    bool merge_with_previous = false;

    for (SegmentList::const_iterator f_it = filter_segmentation.Begin();
          f_it != filter_segmentation.End(); ++f_it) {

      if (merge_with_previous) {
        // Make preceeding segment overlap this segment.
        out_segmentation->Back().end_frame = f_it->end_frame;
      }

      if (f_it->Label() == secondary_label) {
        // This segment must overlap with either previous or next or both
        // segments.
        if (f_it == filter_segmentation.Begin())
          merge_with_previous = false;    // no preceeding segment to overlap
        else if (!merge_with_previous)    // no overlap done with preceeding segment
          merge_with_previous = true;     // so prepare to overlap with preceeding segment
          
        if (merge_with_previous) {
          // Make preceeding segment overlap this segment.
          out_segmentation->Back().end_frame = f_it->end_frame;
        }

        out_segmentation->EmplaceBack(f_it->start_frame, f_it->end_frame, 
                                      it->Label());
        merge_with_previous = true;     // Prepare to overlap with succeeding segment in the next iteration
      } else {
        if (!merge_with_previous)
          out_segmentation->EmplaceBack(f_it->start_frame, f_it->end_frame, 
                                        it->Label());
        merge_with_previous = false;
      }
    }
  }
}

}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Create overlapped sub-segmentation of a segmentation by intersecting with "
        "segments from another segmentation. The regions where the other "
        "segmentation has label 'filter-label' are overlapped with neighbors.\n"
        "\n"
        "Usage: segmentation-create-overlapped-subsegments [options] <segmentation-rspecifier|segmentation-rxfilename> <filter-segmentation-rspecifier|filter-segmentation-rxfilename> <segmentation-wspecifier|segmentation-wxfilename>\n"
        " e.g.: segmentation-create-overlapped-subsegments --binary=false --filter-label=1 --subsegment-label=1000 foo bar -\n"
        "       segmentation-create-overlapped-subsegments --filter-label=1 --subsegment-label=1000 ark:1.foo ark:1.bar ark:-\n";
    
    bool binary = true, ignore_missing = true;
    int32 filter_label = -1;
    ParseOptions po(usage);
    
    po.Register("binary", &binary, 
                "Write in binary mode (only relevant if output is a wxfilename)");
    po.Register("filter-label", &filter_label, "The label on which "
                "filtering is done.");
    po.Register("ignore-missing", &ignore_missing, "Ignore missing "
                "segmentations in filter. If this is set true, then the "
                "segmentations with missing key in filter are written "
                "without any modification.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string segmentation_in_fn = po.GetArg(1),
                secondary_segmentation_in_fn = po.GetArg(2),
                segmentation_out_fn = po.GetArg(3);

    // all these "fn"'s are either rspecifiers or filenames.

    bool in_is_rspecifier =
        (ClassifyRspecifier(segmentation_in_fn, NULL, NULL)
         != kNoRspecifier),
        filter_is_rspecifier = 
        (ClassifyRspecifier(secondary_segmentation_in_fn, NULL, NULL)
         != kNoRspecifier),
        out_is_wspecifier =
        (ClassifyWspecifier(segmentation_out_fn, NULL, NULL, NULL)
         != kNoWspecifier);

    if (in_is_rspecifier != out_is_wspecifier || in_is_rspecifier != filter_is_rspecifier)
      KALDI_ERR << "Cannot mix regular files and archives";
    
    int64 num_done = 0, num_err = 0;
    
    if (!in_is_rspecifier) {
      Segmentation segmentation;
      {
        bool binary_in;
        Input ki(segmentation_in_fn, &binary_in);
        segmentation.Read(ki.Stream(), binary_in);
      }
      Segmentation secondary_segmentation;
      {
        bool binary_in;
        Input ki(secondary_segmentation_in_fn, &binary_in);
        secondary_segmentation.Read(ki.Stream(), binary_in);
      }
      
      Segmentation new_segmentation;
      CreateOverlappedSubSegments(segmentation, secondary_segmentation,
                                  filter_label,
                                  &new_segmentation);
      Output ko(segmentation_out_fn, binary);
      new_segmentation.Write(ko.Stream(), binary);

      KALDI_LOG << "Created subsegments of " << segmentation_in_fn 
                << " based on " << secondary_segmentation_in_fn
                << " and wrote to " << segmentation_out_fn;
      return 0;
    } else {
      SegmentationWriter writer(segmentation_out_fn); 
      SequentialSegmentationReader reader(segmentation_in_fn);
      RandomAccessSegmentationReader filter_reader(secondary_segmentation_in_fn);

      for (; !reader.Done(); reader.Next(), num_done++) {
        const Segmentation &segmentation = reader.Value();
        const std::string &key = reader.Key();
        
        if (!filter_reader.HasKey(key)) {
          KALDI_WARN << "Could not find filter segmentation for utterance "   
                     << key;
          if (!ignore_missing) {
            num_err++;
          } else 
            writer.Write(key, segmentation);
          continue;
        }
        const Segmentation &secondary_segmentation = filter_reader.Value(key);
        
        Segmentation new_segmentation;
        CreateOverlappedSubSegments(segmentation, secondary_segmentation,
                                    filter_label, 
                                    &new_segmentation);

        writer.Write(key, new_segmentation);
      }

      KALDI_LOG << "Created subsegments for " << num_done << " segmentations; "
                << "failed with " << num_err << " segmentations";
      
      return ((num_done != 0 && num_err < num_done) ? 0 : 1);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


