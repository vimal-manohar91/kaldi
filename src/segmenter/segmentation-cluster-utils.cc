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
#include "segmenter/segmentation-utils.h"
#include "segmenter/gaussian-stats-clusterable.h"
#include "tree/cluster-utils.h"

namespace kaldi {
  
void SegmentClusteringOptions::Register(OptionsItf *opts) {
  opts->Register("length-tolerance", &length_tolerance,
                 "Tolerance a lenght difference of this many frames "
                 "between segment end and number of feature frames.");
  opts->Register("window-size", &window_size,
                 "Length of window (in number of frames) over which "
                 "Gaussians are estimated.");
  opts->Register("merge-only-overlapping-segments", 
                 &merge_only_overlapping_segments,
                 "Only merge overlapping segments in ClusterAdjacentSegments.");
  opts->Register("statistics-scale", &statistics_scale,
                 "Scale statistics by this factor.");
  gaussian_stats_opts.Register(opts);
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
  
  GaussianStatsClusterable left_gauss(
      feats_.NumCols(), opts_.gaussian_stats_opts);
  GaussianStatsClusterable right_gauss(
      feats_.NumCols(), opts_.gaussian_stats_opts);
  
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
    EstGaussian(left_feats, 1.0, &left_gauss);
    EstGaussian(right_feats, 1.0, &right_gauss);
    
    // Compute the distance between the left window and the right window.
    // If the two windows are very similar, then the distance must be very small.
    // They are likely to be from from the same cluster.
    // If the two windows are very different, then the distance is very large.
    // We add change points at the points where the distance reaches local 
    // maximum.
    dist = left_gauss.Distance(right_gauss);

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

    dist = left_gauss.Distance(right_gauss);
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

class LinearBottomUpClusterer: BottomUpClusterer {
 public:
  LinearBottomUpClusterer(const std::vector<Clusterable*> &points,      
                          BaseFloat max_merge_threshold,
                          int32 min_clust,
                          std::vector<Clusterable*> *clusters_out,
                          std::vector<int32> *assignments_out)
      : BottomUpClusterer(points, max_merge_threshold, min_clust,
                          clusters_out, assignments_out) { }

  BaseFloat Cluster();

 protected:
  /// Sets up distances and queue. 
  /// Distances will be set only for the adjacent clusters. Only these will
  /// be added to the queue.
  void SetInitialDistances();
  
  /// Reconstructs the priority queue from the distances. Again only the 
  /// adjacent clusters will be added.
  void ReconstructQueue();

  void SetDistance(int32 i, int32 j);
  
  /// Merge j into i and delete j.
  /// Also updates the distance from the cluster i to its adjacent clusters.
  void MergeClusters(int32 i, int32 j);
};

BaseFloat LinearBottomUpClusterer::Cluster() {
  KALDI_VLOG(2) << "Initializing cluster assignments.";
  InitializeAssignments();
  KALDI_VLOG(2) << "Setting initial distances.";
  SetInitialDistances();

  KALDI_VLOG(2) << "Clustering...";
  while (!StoppingCriterion()) {
    std::pair<BaseFloat, std::pair<uint_smaller, uint_smaller> > pr = queue_.top();
    BaseFloat dist = pr.first;
    int32 i = (int32) pr.second.first, j = (int32) pr.second.second;
    queue_.pop();
    if (CanMerge(i, j, dist)) {
      UpdateClustererStats(i, j);
      MergeClusters(i, j);
    }
  }
  KALDI_VLOG(2) << "Renumbering clusters to contiguous numbers.";
  Renumber();
  return ans_;
}

void LinearBottomUpClusterer::SetInitialDistances() {
  for (int32 i = 1; i < npoints_; i++) {
    BaseFloat dist = ComputeDistance(i, i - 1);
    PossiblyConsiderForMerging(i, i - 1);
    KALDI_VLOG(2) << "Distance(" << i << ", " << i - 1 << ") = " << dist;
  }
}

void LinearBottomUpClusterer::MergeClusters(int32 i, int32 j) {
  KALDI_ASSERT(i != j && i < npoints_ && j < npoints_);
  (*clusters_)[i]->Add(*((*clusters_)[j]));
  delete (*clusters_)[j];
  (*clusters_)[j] = NULL;
  // note that we may have to follow the chain within "assignment_" to get
  // final assignments.
  (*assignments_)[j] = i;
  // subtract negated objective function change, i.e. add objective function
  // change.
  ans_ -= dist_vec_[(i * (i - 1)) / 2 + j];
  nclusters_--;
  // Now update "distances".
  for (int32 k = i - 1; k >= 0; k--) {
    if ((*clusters_)[k] != NULL) {
      SetDistance(i, k);  // SetDistance requires k < i.
      break;
    }
  }
  for (int32 k = i + 1; k < npoints_; k++) {
    if ((*clusters_)[k] != NULL) {
      SetDistance(k, i);
      break;
    }
  }
}

void LinearBottomUpClusterer::ReconstructQueue() {
  // empty queue [since there is no clear()]
  {
    QueueType tmp;
    std::swap(tmp, queue_);
  }
  for (int32 i = 1; i < npoints_; i++) {
    if ((*clusters_)[i] != NULL) {
      for (int32 j = i - 1; j >= 0; j--) {
        if ((*clusters_)[j] != NULL) {
          PossiblyConsiderForMerging(i, j);
          break;
        }
      }
    }
  }
}

void LinearBottomUpClusterer::SetDistance(int32 i, int32 j) {
  KALDI_ASSERT(i < npoints_ && j < i && (*clusters_)[i] != NULL
         && (*clusters_)[j] != NULL);
  BaseFloat dist = ComputeDistance(i, j);
  if (GetVerboseLevel() >= 5) {
    std::ostringstream oss_i;
    (*clusters_)[i]->Write(oss_i, false);
    std::ostringstream oss_j;
    (*clusters_)[j]->Write(oss_j, false);

    KALDI_VLOG(5) << "Distance " 
      << i << " (" << oss_i.str() << ") and "
      << j << " (" << oss_j.str() << ") "
      << " = " << dist;
  }
  PossiblyConsiderForMerging(i, j);
  // every time it's at least twice the maximum possible size.
  if (queue_.size() >= static_cast<size_t> (npoints_)) {
    // Control memory use by getting rid of orphaned queue entries
    ReconstructQueue();
  }
}

int32 ClusterAdjacentSegments(const SegmentClusteringOptions &opts,
                              const MatrixBase<BaseFloat> &feats,
                              segmenter::Segmentation *segmentation) {
  if (segmentation->Dim() == 0) {
    return 0;
  }

  if (segmentation->Dim() == 1) {
    return 1;
  }
  
  segmentation->Sort();
  std::vector<Clusterable*> points;
  for (segmenter::SegmentList::const_iterator it = segmentation->Begin();
       it != segmentation->End(); ++it) {
    GaussianStatsClusterable* this_gauss = new GaussianStatsClusterable(
        feats.NumCols(), opts.gaussian_stats_opts);
    int32 end = std::min(it->end_frame, feats.NumRows());
    SubMatrix<BaseFloat> this_feats(
        feats, it->start_frame, end - it->start_frame + 1,
        0, feats.NumCols());
    EstGaussian(this_feats, opts.statistics_scale, this_gauss);
    points.push_back(this_gauss);
  }

  std::vector<int32> assignments_out;
  LinearBottomUpClusterer clusterer(
      points, (opts.gaussian_stats_opts.distance_metric == "bic" ? 
               0.0 : opts.gaussian_stats_opts.threshold),
      1, NULL, &assignments_out);
  clusterer.Cluster();
  DeletePointers(&points);

  int32 i = 0, max_label = 0;
  for (segmenter::SegmentList::iterator it = segmentation->Begin();
       it != segmentation->End(); ++it, i++) {
    it->SetLabel(assignments_out[i] + 1);
    if (assignments_out[i] > max_label) max_label = assignments_out[i];
  }

  return max_label;
}

}  // end namespace kaldi
