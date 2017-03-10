// segmenter/segmentation-cluster-utils.cc

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
#include "segmenter/segmentation.h"
#include "segmenter/gaussian-stats.h"

namespace kaldi {
  
void SegmentClusteringOptions::Register(OptionsItf *opts) {
  opts->Register("length-tolerance", &length_tolerance,
                 "Tolerance a lenght difference of this many frames "
                 "between segment end and number of feature frames.");
  opts->Register("window-size", &window_size,
                 "Length of window (in number of frames) over which "
                 "Gaussians are estimated.");
  opts->Register("use-full-covar", &use_full_covar,
                 "Use full covariance Gaussians.");
  opts->Register("distance-metric", &distance_metric,
                 "Choose a distance metric among kl2 | glr | bic");
  opts->Register("bic-penalty", &bic_penalty,
                 "The lambda term in BIC equation that penalizes model "
                 "complexity.");
  opts->Register("variance-floor", &var_floor,
                 "Floor variances during Gaussian estimation.");
  opts->Register("threshold", &threshold, 
                 "Threshold for merging or splitting segments.");
  opts->Register("merge-only-overlapping-segments", 
                 &merge_only_overlapping_segments,
                 "Only merge overlapping segments in ClusterAdjacentSegments.");
}

BaseFloat Distance(const SegmentClusteringOptions &opts,
                   const GaussianStats &stats1, const GaussianStats &stats2) {
  if (opts.distance_metric == "kl2") {
    return DistanceKL2Diag(stats1, stats2, opts.var_floor);
  } else if (opts.distance_metric == "glr") {
    return DistanceGLR(stats1, stats2, opts.var_floor);
  } else if (opts.distance_metric == "bic") {
    return DistanceBIC(stats1, stats2, opts.bic_penalty, opts.var_floor);
  } else {
    KALDI_ERR << "Unknown distance metric " << opts.distance_metric;
  }

  return -std::numeric_limits<BaseFloat>::max();
}

class SegmentSplitter {
 public: 
  SegmentSplitter(const SegmentClusteringOptions &opts, 
                  const MatrixBase<BaseFloat> &feats,
                  const segmenter::Segmentation &segmentation,
                  segmenter::Segmentation *out_segmentation);

  void Split();

  int32 NumClusters() const { return nclusters_; }
 
 private:
  void SplitSegment(const segmenter::Segment &seg,
                    const std::vector<BaseFloat> &distances);

  void PossiblySplitSegment(int32 offset,
      std::vector<BaseFloat>::const_iterator offset_it,
      std::vector<BaseFloat>::iterator beg_it,
      std::vector<BaseFloat>::iterator end_it,
      int32 min_length);

  void ComputeDistances(const segmenter::Segment &seg,
                        std::vector<BaseFloat> *distances) const;

  const SegmentClusteringOptions &opts_;
  const MatrixBase<BaseFloat> &feats_;
  const segmenter::Segmentation &segmentation_;
  segmenter::Segmentation *out_segmentation_;

  int32 nclusters_;
};

SegmentSplitter::SegmentSplitter(const SegmentClusteringOptions &opts,
                                 const MatrixBase<BaseFloat> &feats,
                                 const segmenter::Segmentation &segmentation,
                                 segmenter::Segmentation *out_segmentation)
  : opts_(opts), feats_(feats), segmentation_(segmentation),
    out_segmentation_(out_segmentation), nclusters_(0) { }

void SegmentSplitter::Split() {
  for (segmenter::SegmentList::const_iterator it = segmentation_.Begin();
       it != segmentation_.End(); ++it) {
    segmenter::Segment seg = *it;
    if (it->end_frame > feats_.NumRows() + opts_.length_tolerance) {
      KALDI_WARN << "Segment with segment end ( " << *it << " ) "
                 << "> number of feature frames; "
                 << feats_.NumRows();
      seg.end_frame = feats_.NumRows() - 1;
    }

    if (seg.Length() <= 2 * opts_.window_size) {
      out_segmentation_->EmplaceBack(seg.start_frame, seg.end_frame,
                                     ++nclusters_);
      continue;
    }

    std::vector<BaseFloat> distances;
    ComputeDistances(seg, &distances);
    SplitSegment(seg, distances);
    //PossiblySplitSegment(it->start_frame, distances.begin(), 
    //                     distances.begin(), distances.end(),
    //                     opts_.window_size);
  }
}

void SegmentSplitter::ComputeDistances(
    const segmenter::Segment &seg,
    std::vector<BaseFloat> *distances) const {
  KALDI_ASSERT(distances);
  distances->clear();
  distances->resize(
      std::min(seg.end_frame + 1, feats_.NumRows()) - seg.start_frame,
      -std::numeric_limits<BaseFloat>::max());
  
  GaussianStats left_gauss(feats_.NumCols(), opts_.use_full_covar);
  GaussianStats right_gauss(feats_.NumCols(), opts_.use_full_covar);
  
  int32 mid = 0;
  BaseFloat dist = -std::numeric_limits<BaseFloat>::max();
  
  {
    // Left window is [s, s + W)
    // Right window is [s + W, s + 2W)
    int32 start = seg.start_frame;
    mid = start + opts_.window_size;
  
    int32 end = mid + opts_.window_size;
    // Adjust the end if it is beyond the end of the feature matrix
    if (end > feats_.NumRows()) end = feats_.NumRows(); 

    SubMatrix<BaseFloat> left_feats(feats_, start, mid - start, 
        0, feats_.NumCols());
    SubMatrix<BaseFloat> right_feats(feats_, mid, end - mid,
        0, feats_.NumCols());

    // Estimate Gaussians for the left window and the right window
    EstGaussian(left_feats, &left_gauss);
    EstGaussian(right_feats, &right_gauss);
    
    // Compute the distance between the left window and the right window.
    // If the two windows are very similar, then the distance must be very small.
    // They are likely to be from from the same cluster.
    // If the two windows are very different, then the distance is very large.
    // We add change points at the points where the distance reaches local 
    // maximum.
    dist = Distance(opts_, left_gauss, right_gauss);

    // Use the same distance for the all the frames in the first window.
    for (int32 i = seg.start_frame; i <= mid; i++) {
      (*distances)[i - seg.start_frame] = dist;
    }
  }

  // Shift the windows frame-by-frame to the right until there is only 
  // opts_.window_size frames left in the right window.
  // Instead of recomputing the Gaussian stats, we will remove the stats from
  // frame in the left and add one from the right.
  for (mid = mid + 1; mid <= seg.end_frame - opts_.window_size; mid++) {
    // Left window is from [mid - W, mid)
    int32 start = mid - opts_.window_size;
    left_gauss.AddStats(feats_.Row(start - 1), -1.0);
    left_gauss.AddStats(feats_.Row(mid), 1.0);

    // Right window is from [mid, mid + W)
    right_gauss.AddStats(feats_.Row(mid), -1.0);

    int32 end = mid + opts_.window_size;
    if (end > std::min(seg.end_frame, feats_.NumRows() - 1)) {
      // If the last frame is beyond the features matrix or the segment end,
      // then we don't have any stats to add.
      end = std::min(seg.end_frame, feats_.NumRows() - 1);
    } else {
      right_gauss.AddStats(feats_.Row(end), 1.0);
    }

    dist = Distance(opts_, left_gauss, right_gauss);
    (*distances)[mid - seg.start_frame] = dist;
  }

  // Use the same distances for all the frame in the last window.
  for (; mid <= std::min(seg.end_frame, feats_.NumRows() - 1); mid++) {
    (*distances)[mid - seg.start_frame] = dist;
  }

  KALDI_ASSERT(mid - seg.start_frame == distances->size());
}

void SegmentSplitter::SplitSegment(const segmenter::Segment &seg,
                                   const std::vector<BaseFloat> &distances) {
  int32 mid = opts_.window_size;
  int32 prev_boundary = 0;
 
  while (mid < distances.size() - opts_.window_size) {
    BaseFloat max = distances[mid - opts_.window_size];
    int32 max_element = mid - opts_.window_size;
    for (int32 i = mid - opts_.window_size; i < mid + opts_.window_size; i++) {
      if (i != mid && distances[i] > max) {
        max = distances[i];
        max_element = i;
      }
    }

    if (distances[mid] > max) {
      // mid frame is a local maximum. 
      // End the previous segment at this point.
      out_segmentation_->EmplaceBack(seg.start_frame + prev_boundary, 
                                     seg.start_frame + mid - 1, ++nclusters_);
      prev_boundary = mid;
      mid += opts_.window_size;
    } else {
      // mid frame is not a local maximum. Continue searching for it.
      if (max_element > mid) {
        // Move "mid" to the previously found max_element.
        mid = max_element;
      } else {
        mid++;
      }
    }
  } 

  {
    int32 end = std::min(
        seg.end_frame, 
        seg.start_frame + static_cast<int32>(distances.size()) - 1);
    out_segmentation_->EmplaceBack(
        seg.start_frame + prev_boundary, end,
        ++nclusters_);
  }
}

void SegmentSplitter::PossiblySplitSegment(
    int32 offset, std::vector<BaseFloat>::const_iterator offset_it,
    std::vector<BaseFloat>::iterator beg_it,
    std::vector<BaseFloat>::iterator end_it,
    int32 min_length) {
  if (static_cast<int32>(end_it - beg_it) <= 2 * min_length + 1) {
    // This segment is too short to be split. So return it as a full segment.
    segmenter::Segment seg;
    seg.start_frame = static_cast<int32>(beg_it - offset_it) + offset;
    seg.end_frame = static_cast<int32>(end_it - offset_it) + offset;
    seg.SetLabel(++nclusters_);
    out_segmentation_->PushBack(seg);
  } else {
    // Choose a local maximum as a split point such that splitting it 
    // at that point will leave at least min_length frames on both sides.
    std::vector<BaseFloat>::iterator max_it = std::max_element(
        beg_it + min_length, end_it - min_length);

    PossiblySplitSegment(offset, offset_it, beg_it, max_it, min_length);
    PossiblySplitSegment(offset, offset_it, max_it, end_it, min_length);
  }
}

int32 SplitByChangePoints(const SegmentClusteringOptions &opts,
                          const MatrixBase<BaseFloat> &feats,
                          const segmenter::Segmentation &segmentation,
                          segmenter::Segmentation *out_segmentation) {
  SegmentSplitter splitter(opts, feats, segmentation,
                           out_segmentation);
  splitter.Split();
  return splitter.NumClusters();
}

class AdjacentSegmentClusterer {
 public:
  AdjacentSegmentClusterer(const SegmentClusteringOptions &opts,
                           const MatrixBase<BaseFloat> &feats,
                           segmenter::Segmentation *segmentation)
    : opts_(opts), feats_(feats), segmentation_(segmentation),
      nclusters_(0) { segmentation_->Sort(); }

  void Cluster();
  
  int32 NumClusters() const { return nclusters_; }
  
 private:
  const SegmentClusteringOptions &opts_;
  const MatrixBase<BaseFloat> &feats_;
  
  segmenter::Segmentation *segmentation_;

  int32 nclusters_;
};

void AdjacentSegmentClusterer::Cluster() {
  if (segmentation_->Dim() == 0) {
    return;
  }

  if (segmentation_->Dim() == 1) {
    nclusters_ = 1;
    return;
  }
  
  segmenter::SegmentList::iterator it = segmentation_->Begin(), 
    next_it = segmentation_->Begin();
  ++next_it;

  nclusters_ = 1;

  GaussianStats this_gauss(feats_.NumCols(), opts_.use_full_covar);
  {
    int32 end = std::min(it->end_frame, feats_.NumRows());
    SubMatrix<BaseFloat> this_feats(
        feats_, it->start_frame, end - it->start_frame + 1,
        0, feats_.NumCols());

    EstGaussian(this_feats, &this_gauss);
  }
  for (; next_it != segmentation_->End(); ++it, ++next_it) {
    segmenter::SegmentList::iterator test_it = it;
    ++test_it;
    KALDI_ASSERT(test_it == next_it);
    if (!opts_.merge_only_overlapping_segments || 
        next_it->start_frame <= it->end_frame) {
      // Consider merging segments *it and *next_it
      GaussianStats next_gauss(feats_.NumCols(), opts_.use_full_covar);
      {
        int32 next_end = std::min(next_it->end_frame, feats_.NumRows());
        SubMatrix<BaseFloat> next_feats(
            feats_, next_it->start_frame, next_end - next_it->start_frame + 1,
            0, feats_.NumCols());

        EstGaussian(next_feats, &next_gauss);
      }
      BaseFloat dist = Distance(opts_, this_gauss, next_gauss);

      if (dist < opts_.threshold) {
        // Merge segments
        if (next_it->start_frame <= it->end_frame + 1) {
          it->end_frame = next_it->end_frame;
          next_it = segmentation_->Erase(next_it);
          if (next_it == segmentation_->End()) break;
        } else {
          next_it->SetLabel(it->Label());
        }
      } else {
        // Segments not merged
        nclusters_++;
      }
      this_gauss = next_gauss;
    } else {
      {
        int32 next_end = std::min(next_it->end_frame, feats_.NumRows());
        SubMatrix<BaseFloat> next_feats(
            feats_, next_it->start_frame, next_end - next_it->start_frame + 1,
            0, feats_.NumCols());

        EstGaussian(next_feats, &this_gauss);
      }
      nclusters_++;
    }
  }
  KALDI_ASSERT(segmentation_->Dim() >= 1);
}

int32 ClusterAdjacentSegments(const SegmentClusteringOptions &opts,
                              const MatrixBase<BaseFloat> &feats,
                              segmenter::Segmentation *segmentation) {
  AdjacentSegmentClusterer sc(opts, feats, segmentation);
  sc.Cluster();
  return sc.NumClusters();
}

}  // end namespace kaldi
