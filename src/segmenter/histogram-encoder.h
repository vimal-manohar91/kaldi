#ifndef KALDI_SEGMENTER_HISTOGRAM_ENCODER_H_
#define KALDI_SEGMENTER_HISTOGRAM_ENCODER_H_

#include "base/kaldi-common.h"
#include "matrix/kaldi-matrix.h"
#include "itf/options-itf.h"

namespace kaldi {
namespace segmenter {

/** This structure is used to encode some vector of real values into bins. This
 *  is mainly used in the classification of segments into speech, silence and
 *  noise depending on the vector of frame-level energy and/or zero-crossing
 *  of the frames in the segment.
**/

struct HistogramEncoder {
  // Width of the bins in the histogram of real values
  BaseFloat bin_width;

  // Minimum score corresponding to the lowest bin of the histogram
  BaseFloat min_score; 

  // This is a vector that stores the number of real values contained in the
  // different bins.
  std::vector<int32> bin_sizes;

  // A flag that is relevant only in a particular function. See the comments 
  // in Encode function for details.
  bool select_from_full_histogram;

  // default constructor
  HistogramEncoder(): bin_width(-1), 
                      min_score(std::numeric_limits<BaseFloat>::infinity()),
                      select_from_full_histogram(false) {}

  // Accessors for different quantities
  inline int32 NumBins() const { return bin_sizes.size(); } 
  inline int32 BinSize(int32 i) const { return bin_sizes[i]; }

  // Initialize the container to a specific number of bins and also size 
  // and the value each bin represents.
  void Initialize(int32 num_bins, BaseFloat bin_w, BaseFloat min_s);

  // Insert the real value 'x' with a count of 'n' times into the appropriate
  // bin in the histogram.
  void Encode(BaseFloat x, int32 n);
};

/** 
 * Structure for options for histogram encoding 
**/

struct HistogramOptions {
  int32 num_bins;
  bool select_above_mean;
  bool select_from_full_histogram;

  HistogramOptions() : num_bins(100), select_above_mean(false), select_from_full_histogram(false) {}
  
  void Register(OptionsItf *opts) {
    opts->Register("num-bins", &num_bins, "Number of bins in the histogram "
                   "created using the scores. Use larger number of bins to "
                   "make a finer selection");
    opts->Register("select-above-mean", &select_above_mean, "If true, "
                   "use mean as the reference instead of min");
    opts->Register("select-from-full-histogram", &select_from_full_histogram,
                   "Do not restrict selection to one half");

  }

};

    /***
     * DEPRECATED SAD FUCTIONS

    // Create a Histogram Encoder that can map a segment to 
    // a bin based on the average score
    void CreateHistogram(int32 label, const Vector<BaseFloat> &score, 
                         const HistogramOptions &opts, HistogramEncoder *hist);

    // Modify this segmentation to select the top bins in the 
    // histogram. Assumes that this segmentation also has the 
    // average scores.
    int32 SelectTopBins(const HistogramEncoder &hist, 
                        int32 src_label, int32 dst_label, int32 reject_label,
                        int32 num_frames_select, bool remove_rejected_frames);

    // Modify this segmentation to select the bottom bins in the histogram.
    // Assumes that this segmentation also has the average scores.
    int32 SelectBottomBins(const HistogramEncoder &hist, 
                           int32 src_label, int32 dst_label, int32 reject_label,
                           int32 num_frames_select, bool remove_rejected_frames);

    // Modify this segmentation to select the top and bottom bins in the 
    // histogram. Assumes that this segmentation also has the average scores.
    std::pair<int32,int32> SelectTopAndBottomBins(
        const HistogramEncoder &hist_encoder, 
        int32 src_label, int32 top_label, int32 num_frames_top,
        int32 bottom_label, int32 num_frames_bottom,
        int32 reject_label, bool remove_rejected_frames);
     
     * END DEPRECATED SAD FUCTIONS
     **/

#endif // KALDI_SEGMENTER_HISTOGRAM_ENCODER_H_
