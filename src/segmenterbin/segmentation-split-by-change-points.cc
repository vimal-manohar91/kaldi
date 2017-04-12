// segmenterbin/segmentation-split-by-change-points.cc

// Copyright 2017   Vimal Manohar (Johns Hopkins University)

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
#include "segmenter/segmentation-cluster-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Splits a segmentation by detecting change points "
        "using KL2 (symmetric KL divergence) distance between "
        "adjacent Gaussian windows.\n"
        "Usage: segmentation-split-by-change-points [options] "
        "<segmentation-rspecifier> <feats-rspecifier> <segmentation-wspecifier>\n"
        " e.g.: segmentation-split-by-change-points ark:foo.seg ark:feats.ark ark,t:-\n"
        "See also: segmentation-split-segments, "
        "segmentation-post-process --max-segment-length\n";

    bool binary = true;

    ParseOptions po(usage);

    po.Register("binary", &binary,
                "Write in binary mode "
                "(only relevant if output is a wxfilename)");
    
    SegmentClusteringOptions opts;
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string segmentation_in_fn = po.GetArg(1),
      feats_in_fn = po.GetArg(2),
      segmentation_out_fn = po.GetArg(3);

    // all these "fn"'s are either rspecifiers or filenames.
    bool in_is_rspecifier =
        (ClassifyRspecifier(segmentation_in_fn, NULL, NULL)
         != kNoRspecifier),
        out_is_wspecifier =
        (ClassifyWspecifier(segmentation_out_fn, NULL, NULL, NULL)
         != kNoWspecifier);
    
    if (in_is_rspecifier != out_is_wspecifier)
      KALDI_ERR << "Cannot mix regular files and archives";

    if (!in_is_rspecifier) {
      Segmentation segmentation;
      ReadKaldiObject(segmentation_in_fn, &segmentation);

      Matrix<BaseFloat> feats;
      ReadKaldiObject(feats_in_fn, &feats);

      Segmentation out_segmentation;
      int32 num_clusters = SplitByChangePoints(opts, feats, segmentation, 
                                               &out_segmentation);
      
      KALDI_LOG << "Clustered segments; got " << num_clusters << " clusters.";
      WriteKaldiObject(segmentation, segmentation_out_fn, binary);

      return 0;
    } else {
      int32 num_done = 0, num_err = 0;

      SequentialSegmentationReader segmentation_reader(segmentation_in_fn);
      RandomAccessBaseFloatMatrixReader feats_reader(feats_in_fn);
      SegmentationWriter segmentation_writer(segmentation_out_fn);

      for (; !segmentation_reader.Done(); segmentation_reader.Next()) {
        const Segmentation &segmentation = segmentation_reader.Value();
        const std::string &key = segmentation_reader.Key();

        if (!feats_reader.HasKey(key)) {
          KALDI_WARN << "Could not find key " << key << " in " 
                     << "feats-rspecifier " << feats_in_fn;
          num_err++;
          continue;
        }

        const MatrixBase<BaseFloat> &feats = feats_reader.Value(key);
        
        Segmentation out_segmentation;
        int32 num_clusters = SplitByChangePoints(opts, feats, segmentation, 
                                                 &out_segmentation);

        KALDI_VLOG(2) << "For key " << key << ", got " << num_clusters
                      << " segments.";

        segmentation_writer.Write(key, out_segmentation);
        num_done++;
      }

      KALDI_LOG << "Clustered segments from " << num_done << " recordings "
                << "failed with " << num_err;
      return (num_done != 0 ? 0 : 1);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
