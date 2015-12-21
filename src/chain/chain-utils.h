namespace kaldi {
namespace chain {

// writes compressed as unsigned char a vector 'vec' that is required to have
// values between 0 and 1.
static inline void WriteVectorAsChar(std::ostream &os,
                                     bool binary,
                                     const VectorBase<BaseFloat> &vec);

// reads data written by WriteVectorAsChar.
static inline void ReadVectorAsChar(std::istream &is,
                                    bool binary,
                                    Vector<BaseFloat> *vec);

void RoundUpNumFrames(int32 frame_subsampling_factor,
                      int32 *num_frames,
                      int32 *num_frames_overlap);

/// This function helps you to pseudo-randomly split a sequence of length 'num_frames',
/// interpreted as frames 0 ... num_frames - 1, into pieces of length exactly
/// 'frames_per_range', to be used as examples for training.  Because frames_per_range
/// may not exactly divide 'num_frames', this function will leave either small gaps or
/// small overlaps in pseudo-random places.
/// The output 'range_starts' will be set to a list of the starts of ranges, the
/// output ranges are of the form
/// [ (*range_starts)[i] ... (*range_starts)[i] + frames_per_range - 1 ].
void SplitIntoRanges(int32 num_frames,
                     int32 frames_per_range,
                     std::vector<int32> *range_starts);


/// This utility function is not used directly in the 'chain' code.  It is used
/// to get weights for the derivatives, so that we don't doubly train on some
/// frames after splitting them up into overlapping ranges of frames.  The input
/// 'range_starts' will be obtained from 'SplitIntoRanges', but the
/// 'range_length', which is a length in frames, may be longer than the one
/// supplied to SplitIntoRanges, due the 'overlap'.  (see the calling code...
/// if we want overlapping ranges, we get it by 'faking' the input to
/// SplitIntoRanges).
///
/// The output vector 'weights' will be given the same dimension as
/// 'range_starts'.  By default the output weights in '*weights' will be vectors
/// of all ones, of length equal to 'range_length', and '(*weights)[i]' represents
/// the weights given to frames numbered
///   t = range_starts[i] ... range_starts[i] + range_length - 1.
/// If these ranges for two successive 'i' values overlap, then we
/// reduce the weights to ensure that no 't' value gets a total weight
/// greater than 1.  We do this by dividing the overlapped region
/// into three approximately equal parts, and giving the left part
/// to the left range; the right part to the right range; and
/// in between, interpolating linearly.
void GetWeightsForRanges(int32 range_length,
                         const std::vector<int32> &range_starts,
                         std::vector<Vector<BaseFloat> > *weights);


} // namespace chain
} // namespace kaldi
