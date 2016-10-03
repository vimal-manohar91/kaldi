#include "segmenter/histogram-encoder.h"

namespace kaldi {
namespace segmenter {

/**
 * DEPRECATED Histogram FUCTIONS
 *
void HistogramEncoder::Initialize(int32 num_bins, BaseFloat bin_w, BaseFloat min_s) {
  bin_sizes.clear();
  bin_sizes.resize(num_bins, 0);
  bin_width = bin_w;
  min_score = min_s;
}

void HistogramEncoder::Encode(BaseFloat x, int32 n) {
  int32 i = (x - min_score ) / bin_width;
  if (i < 0) i = 0;
  if (i >= NumBins()) i = NumBins() - 1;
  bin_sizes[i] += n;
}
 * END DEPRECATED Histogram Functions
 * */

/**
 * DEPRECATED SAD FUNCTIONS

 // Create a HistogramEncoder object based on this segmentation
void Segmentation::CreateHistogram(
    int32 label, const Vector<BaseFloat> &scores, 
    const HistogramOptions &opts, HistogramEncoder *hist_encoder) {
  if (Dim() == 0)
    KALDI_ERR << "Segmentation must not be empty";

  int32 num_bins = opts.num_bins;
  BaseFloat min_score = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat max_score = -std::numeric_limits<BaseFloat>::infinity();

  mean_scores_.clear();
  mean_scores_.resize(Dim(), std::numeric_limits<BaseFloat>::quiet_NaN());
  
  std::vector<int32> num_frames(Dim(), 0);

  int32 i = 0;
  for (SegmentList::iterator it = segments_.begin(); 
        it != segments_.end(); ++it, i++) {
    if (it->Label() != label) continue;
    SubVector<BaseFloat> this_segment_scores(scores, it->start_frame, it->end_frame - it->start_frame + 1);
    BaseFloat mean_score = this_segment_scores.Sum() / this_segment_scores.Dim();
    
    mean_scores_[i] = mean_score;
    num_frames[i] = this_segment_scores.Dim();

    if (mean_score > max_score) max_score = mean_score;
    if (mean_score < min_score) min_score = mean_score;
  }
  KALDI_ASSERT(i == mean_scores_.size());

  if (opts.select_above_mean) {
    min_score = scores.Sum() / scores.Dim();
  }

  BaseFloat bin_width = (max_score - min_score) / num_bins;
  hist_encoder->Initialize(num_bins, bin_width, min_score);

  hist_encoder->select_from_full_histogram = opts.select_from_full_histogram;

  i = 0;
  for (SegmentList::const_iterator it = segments_.begin(); it != segments_.end(); ++it, i++) {
    if (it->Label() != label) continue;
    KALDI_ASSERT(!KALDI_ISNAN(mean_scores_[i]));

    if (opts.select_above_mean && mean_scores_[i] < min_score) continue;
    KALDI_ASSERT(mean_scores_[i] >= min_score);

    hist_encoder->Encode(mean_scores_[i], num_frames[i]);
  }
  KALDI_ASSERT(i == mean_scores_.size());
  Check();
}

int32 Segmentation::SelectTopBins(
    const HistogramEncoder &hist_encoder, 
    int32 src_label, int32 dst_label, int32 reject_label,
    int32 num_frames_select, bool remove_rejected_frames) {
  KALDI_ASSERT(mean_scores_.size() == Dim());
  KALDI_ASSERT(dst_label >=0 && reject_label >= 0);
  KALDI_ASSERT(num_frames_select >= 0);

  BaseFloat min_score_for_selection = std::numeric_limits<BaseFloat>::infinity();
  int32 num_top_frames = 0, i = hist_encoder.NumBins() - 1;
  while (i >= (hist_encoder.select_from_full_histogram ? 0 : (hist_encoder.NumBins() / 2))) {
    num_top_frames += hist_encoder.BinSize(i);
    if (num_top_frames >= num_frames_select) {
      num_top_frames -= hist_encoder.BinSize(i);
      if (num_top_frames == 0) {
        num_top_frames += hist_encoder.BinSize(i);
        i--;
      }
      break;
    }
    i--;
  }
  min_score_for_selection = hist_encoder.min_score + (i+1) * hist_encoder.bin_width;

  i = 0;
  for (SegmentList::iterator it = segments_.begin(); 
        it != segments_.end(); i++) {
    if (it->Label() != src_label) {
      ++it;
      continue;
    }
    KALDI_ASSERT(!KALDI_ISNAN(mean_scores_[i]));
    if (mean_scores_[i] >= min_score_for_selection) {
      it->SetLabel(dst_label);
      ++it;
    } else {
      if (remove_rejected_frames) {
        it = segments_.erase(it);
        dim_--;
      } else {
        it->SetLabel(reject_label);
        ++it;
      }
    }
  }
  KALDI_ASSERT(i == mean_scores_.size());

  if (remove_rejected_frames) mean_scores_.clear();

  Check();
  return num_top_frames;
}

int32 Segmentation::SelectBottomBins(
    const HistogramEncoder &hist_encoder, 
    int32 src_label, int32 dst_label, int32 reject_label, 
    int32 num_frames_select, bool remove_rejected_frames) {
  KALDI_ASSERT(mean_scores_.size() == Dim());
  KALDI_ASSERT(dst_label >=0 && reject_label >= 0);
  KALDI_ASSERT(num_frames_select >= 0);

  BaseFloat max_score_for_selection = -std::numeric_limits<BaseFloat>::infinity();
  int32 num_bottom_frames = 0, i = 0;
  while (i < (hist_encoder.select_from_full_histogram ? hist_encoder.NumBins() : (hist_encoder.NumBins() / 2))) {
    num_bottom_frames += hist_encoder.BinSize(i);
    if (num_bottom_frames >= num_frames_select) {
      num_bottom_frames -= hist_encoder.BinSize(i);
      if (num_bottom_frames == 0) {
        num_bottom_frames += hist_encoder.BinSize(i);
        i++;
      }
      break;
    }
    i++;
  }
  max_score_for_selection = hist_encoder.min_score + i * hist_encoder.bin_width;

  i = 0;
  for (SegmentList::iterator it = segments_.begin(); 
        it != segments_.end(); i++) {
    if (it->Label() != src_label) {
      ++it; 
      continue;
    }
    KALDI_ASSERT(!KALDI_ISNAN(mean_scores_[i]));
    if (mean_scores_[i] < max_score_for_selection) {
      it->SetLabel(dst_label);
      ++it;
    } else {
      if (remove_rejected_frames) {
        it = segments_.erase(it);
        dim_--;
      } else {
        it->SetLabel(reject_label);
        ++it;
      }
    }
  }
  KALDI_ASSERT(i == mean_scores_.size());

  if (remove_rejected_frames) mean_scores_.clear();

  Check();
  return num_bottom_frames;
}

std::pair<int32,int32> Segmentation::SelectTopAndBottomBins(
    const HistogramEncoder &hist_encoder, 
    int32 src_label, int32 top_label, int32 num_frames_top,
    int32 bottom_label, int32 num_frames_bottom,
    int32 reject_label, bool remove_rejected_frames) {
  KALDI_ASSERT(mean_scores_.size() == Dim());
  KALDI_ASSERT(top_label >= 0 && bottom_label >= 0 && reject_label >= 0);
  KALDI_ASSERT(num_frames_top >= 0 && num_frames_bottom >= 0);
  
  BaseFloat min_score_for_selection = std::numeric_limits<BaseFloat>::infinity();
  int32 num_selected_top = 0, i = hist_encoder.NumBins() - 1;
  while (i >= hist_encoder.NumBins() / 2) {
    int32 this_selected = hist_encoder.BinSize(i);
    num_selected_top += this_selected;
    if (num_selected_top >= num_frames_top) {
      num_selected_top -= this_selected;
      if (num_selected_top == 0) {
        num_selected_top += this_selected;
        i--;
      }
      break;
    }
    i--;
  }
  min_score_for_selection = hist_encoder.min_score + (i+1) * hist_encoder.bin_width;
  
  BaseFloat max_score_for_selection = -std::numeric_limits<BaseFloat>::infinity();
  int32 num_selected_bottom= 0;
  i = 0;
  while (i < hist_encoder.NumBins() / 2) {
    int32 this_selected = hist_encoder.BinSize(i);
    num_selected_bottom += this_selected;
    if (num_selected_bottom >= num_frames_bottom) {
      num_selected_bottom -= this_selected;
      if (num_selected_bottom == 0) {
        num_selected_bottom += this_selected;
        i++;
      }
      break;
    }
    i++;
  }
  max_score_for_selection = hist_encoder.min_score + i * hist_encoder.bin_width;

  i = 0;
  for (SegmentList::iterator it = segments_.begin(); 
        it != segments_.end(); i++) {
    if (it->Label() != src_label) {
      ++it; 
      continue;
    }
    KALDI_ASSERT(!KALDI_ISNAN(mean_scores_[i]));
    if (mean_scores_[i] >= min_score_for_selection) {
      it->SetLabel(top_label);
      ++it;
    } else if (mean_scores_[i] < max_score_for_selection) {
      it->SetLabel(bottom_label);
      ++it;
    } else {
      if (remove_rejected_frames) {
        it = segments_.erase(it);
        dim_--;
      } else {
        it->SetLabel(reject_label);
        ++it;
      }
    }
  }
  KALDI_ASSERT(i == mean_scores_.size());
  
  if (remove_rejected_frames) mean_scores_.clear();

  Check();
  return std::make_pair(num_selected_top, num_selected_bottom);
}

 * END DEPRECATED SAD FUNCTIONS
 **/

} // end namespace segmenter
} // end namespace kaldi
