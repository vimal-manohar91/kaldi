#include "chain/chain-utils.h"

namespace kaldi {
namespace chain {

void WriteVectorAsChar(std::ostream &os,
                       bool binary,
                       const VectorBase<BaseFloat> &vec) {
  if (binary) {
    int32 dim = vec.Dim();
    std::vector<unsigned char> char_vec(dim);
    const BaseFloat *data = vec.Data();
    for (int32 i = 0; i < dim; i++) {
      BaseFloat value = data[i];
      KALDI_ASSERT(value >= 0.0 && value <= 1.0);
      // below, the adding 0.5 is done so that we round to the closest integer
      // rather than rounding down (since static_cast will round down).
      char_vec[i] = static_cast<unsigned char>(255.0 * value + 0.5);
    }
    WriteIntegerVector(os, binary, char_vec);
  } else {
    // the regular floating-point format will be more readable for text mode.
    vec.Write(os, binary);
  }
}

void ReadVectorAsChar(std::istream &is,
                      bool binary,
                      Vector<BaseFloat> *vec) {
  if (binary) {
    BaseFloat scale = 1.0 / 255.0;
    std::vector<unsigned char> char_vec;
    ReadIntegerVector(is, binary, &char_vec);
    int32 dim = char_vec.size();
    vec->Resize(dim, kUndefined);
    BaseFloat *data = vec->Data();
    for (int32 i = 0; i < dim; i++)
      data[i] = scale * char_vec[i];
  } else {
    vec->Read(is, binary);
  }
}

void RoundUpNumFrames(int32 frame_subsampling_factor,
                      int32 *num_frames,
                      int32 *num_frames_overlap) {
  if (*num_frames % frame_subsampling_factor != 0) {
    int32 new_num_frames = frame_subsampling_factor *
        (*num_frames / frame_subsampling_factor + 1);
    KALDI_LOG << "Rounding up --num-frames=" << (*num_frames)
              << " to a multiple of --frame-subsampling-factor="
              << frame_subsampling_factor
              << ", now --num-frames=" << new_num_frames;
    *num_frames = new_num_frames;
  }
  if (*num_frames_overlap % frame_subsampling_factor != 0) {
    int32 new_num_frames_overlap = frame_subsampling_factor *
        (*num_frames_overlap / frame_subsampling_factor + 1);
    KALDI_LOG << "Rounding up --num-frames-overlap=" << (*num_frames_overlap)
              << " to a multiple of --frame-subsampling-factor="
              << frame_subsampling_factor
              << ", now --num-frames-overlap=" << new_num_frames_overlap;
    *num_frames_overlap = new_num_frames_overlap;
  }
  if (*num_frames_overlap < 0 || *num_frames_overlap >= *num_frames) {
    KALDI_ERR << "--num-frames-overlap=" << (*num_frames_overlap) << " < "
              << "--num-frames=" << (*num_frames);
  }

}

void SplitIntoRanges(int32 num_frames,
                     int32 frames_per_range,
                     std::vector<int32> *range_starts) {
  if (frames_per_range > num_frames) {
    range_starts->clear();
    return;  // there is no room for even one range.
  }
  int32 num_ranges = num_frames  / frames_per_range,
      extra_frames = num_frames % frames_per_range;
  // this is a kind of heuristic.  If the number of frames we'd
  // be skipping is less than 1/4 of the frames_per_range, then
  // skip frames; otherwise, duplicate frames.
  // it's important that this is <=, not <, so that if
  // extra_frames == 0 and frames_per_range is < 4, we
  // don't insert an extra range.
  if (extra_frames <= frames_per_range / 4) {
    // skip frames.  we do this at start or end, or between ranges.
    std::vector<int32> num_skips(num_ranges + 1, 0);
    for (int32 i = 0; i < extra_frames; i++)
      num_skips[RandInt(0, num_ranges)]++;
    range_starts->resize(num_ranges);
    int32 cur_start = num_skips[0];
    for (int32 i = 0; i < num_ranges; i++) {
      (*range_starts)[i] = cur_start;
      cur_start += frames_per_range;
      cur_start += num_skips[i + 1];
    }
    KALDI_ASSERT(cur_start == num_frames);
  } else {
    // duplicate frames.
    num_ranges++;
    int32 num_duplicated_frames = frames_per_range - extra_frames;
    // the way we handle the 'extra_frames' frames of output is that we
    // backtrack zero or more frames between outputting each pair of ranges, and
    // the total of these backtracks equals 'extra_frames'.
    std::vector<int32> num_backtracks(num_ranges, 0);
    for (int32 i = 0; i < num_duplicated_frames; i++) {
      // num_ranges - 2 below is not a bug.  we only want to backtrack
      // between ranges, not past the end of the last range (i.e. at
      // position num_ranges - 1).  we make the vector one longer to
      // simplify the loop below.
      num_backtracks[RandInt(0, num_ranges - 2)]++;
    }
    range_starts->resize(num_ranges);
    int32 cur_start = 0;
    for (int32 i = 0; i < num_ranges; i++) {
      (*range_starts)[i] = cur_start;
      cur_start += frames_per_range;
      cur_start -= num_backtracks[i];
    }
    KALDI_ASSERT(cur_start == num_frames);
  }
}

} // namespace chain
} // namespace kaldi
