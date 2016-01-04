// segmenterbin/segmentation-post-process.cc

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
        "Remove short segments from the segmentation and merge to neighbors\n"
        "\n"
        "Usage: segmentation-post-process [options] (segmentation-in-rspecifier|segmentation-in-rxfilename) (segmentation-out-wspecifier|segmentation-out-wxfilename)\n"
        " e.g.: segmentation-post-process --binary=false foo -\n"
        "   segmentation-copy ark:1.ali ark,t:-\n";
    
    bool binary = true;
    int32 max_remove_length = -1;
    int32 widen_length = -1;
    int32 widen_label = -1;
    int32 max_segment_length = -1;
    int32 max_intersegment_length = 1;
    int32 overlap_length = 0;
    bool merge_adjacent_segments = false;

    ParseOptions po(usage);
    
    SegmentationOptions opts;

    std::string remove_labels_csl;

    int32 &remove_label = opts.merge_dst_label;

    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");
    po.Register("remove-label", &remove_label, "The label for which the short segments are to be removed. "
                "If --merge-labels is specified, then all of them would be removed instead.");
    po.Register("remove-labels", &remove_labels_csl, 
                "Remove all segments of these labels.");
    po.Register("max-remove-length", &max_remove_length, "The maximum length of segment in number of frames that will be removed");
    po.Register("widen-label", &widen_label, "Widen segments of this label");
    po.Register("widen-length", &widen_length, "Widen by this amount of frames on either sides");
    po.Register("max-segment-length", &max_segment_length, 
                "If segment is longer than this length, split it into "
                "pieces with less than these many frames.");
    po.Register("overlap-length", &overlap_length,
                "When splitting segments longer than max-segment-length, "
                "have the pieces overlap by these many frames");
    po.Register("merge-adjacent-segments", &merge_adjacent_segments, 
                "Merge adjacent segments of the same label if they are within max-intersegment-length distance");
    po.Register("max-intersegment-length", &max_intersegment_length,  
                "The maximum intersegment length that is allowed for two adjacent segments to be merged");

    opts.Register(&po);

    po.Read(argc, argv); 
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    if (merge_adjacent_segments) 
      KALDI_LOG << "Merging adjacent segments...";

    std::vector<int32> merge_labels;
    RandomAccessSegmentationReader filter_reader(opts.filter_rspecifier);

    std::vector<int32> remove_labels;
    if (remove_labels_csl != "") {
      if (!SplitStringToIntegers(remove_labels_csl, ":",
            false, &remove_labels)) {
        KALDI_ERR << "Bad value for --remove-labels option: "
                  << remove_labels_csl;
      }
      std::sort(remove_labels.begin(), remove_labels.end());
    }

    if (opts.merge_labels_csl != "") {
      if (!SplitStringToIntegers(opts.merge_labels_csl, ":", false,
            &merge_labels)) {
        KALDI_ERR << "Bad value for --merge-labels option: "
                  << opts.merge_labels_csl;
      }
      std::sort(merge_labels.begin(), merge_labels.end());
    }

    std::string segmentation_in_fn = po.GetArg(1),
        segmentation_out_fn = po.GetArg(2);

    int64  num_done = 0, num_err = 0;
    
    SegmentationWriter writer(segmentation_out_fn); 
    SequentialSegmentationReader reader(segmentation_in_fn);
    for (; !reader.Done(); reader.Next(), num_done++) {
      Segmentation seg(reader.Value());
      std::string key = reader.Key();

      if (opts.filter_rspecifier != "") {
        if (!filter_reader.HasKey(key)) {
          KALDI_WARN << "Could not find filter for utterance " << key;
          num_err++;
          continue;
        }
        const Segmentation &filter_segmentation = filter_reader.Value(key);
        seg.IntersectSegments(filter_segmentation, opts.filter_label);
      }

      if (opts.merge_labels_csl != "") {
        seg.MergeLabels(merge_labels, opts.merge_dst_label);
      }

      if (widen_length > 0)
        seg.WidenSegments(widen_label, widen_length);
      if (max_remove_length >= 0)
        seg.RemoveShortSegments(opts.merge_dst_label, max_remove_length);

      if (remove_labels_csl != "")
        seg.RemoveSegments(remove_labels);

      if (merge_adjacent_segments)
        seg.MergeAdjacentSegments(max_intersegment_length);

      if (max_segment_length >= 0)
        seg.SplitSegments(max_segment_length, 
                          max_segment_length/2, overlap_length);

      writer.Write(key, seg);
    }

    KALDI_LOG << "Copied " << num_done << " segmentation; failed with "
      << num_err << " segmentations";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

