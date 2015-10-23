// feat/raw-signal.h

// Copyright    2015 Pegah Ghahremani 

#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>

#include "base/kaldi-error.h" 
#include "matrix/matrix-lib.h" 
#include "itf/online-feature-itf.h"
#include "util/common-utils.h" 
#include "feat/resample.h"
#include "feat/feature-functions.h"
#include "feat/online-feature.h"
namespace kaldi {

// This struct used to extract raw frames 
struct RawSignalOptions {
  BaseFloat frame_length_ms; // in milliseconds.
  BaseFloat samp_freq;
  bool remove_dc_offset;
  bool snip_edges;

  RawSignalOptions() : frame_length_ms(10), 
    samp_freq(16000),
    snip_edges(true) { }
  void Register(OptionsItf *po) {
    po->Register("frame-length-ms", &frame_length_ms, 
                 "Frame length of wave-form in millisecond");

    po->Register("sample-frequency", &samp_freq,

                   "Waveform data sample frequency (must match the waveform file, "  
                   "if specified there)"); 
    po->Register("snip-edges", &snip_edges,
                 " ");
  }
  int32 WindowSize() const {
    return static_cast<int32>(samp_freq * 0.001 * frame_length_ms);
  }
};

// This class extracts raw frames and frames do not have overlap. 
class ComputeRawFrames {
  public:
    explicit ComputeRawFrames(const RawSignalOptions &opts) : opts_(opts) { };
    
    void ExtractRawFrames(const VectorBase<BaseFloat> &wave,
                          int32 sample_index, 
                          VectorBase<BaseFloat> *window) const;
    void AcceptWaveForm(const VectorBase<BaseFloat> &Wave);

    void InputFinished();
    
    int32 Dim() { return opts_.WindowSize(); }

    int32 NumFramesReady() const { return frames_info_.size(); }

    void GetFrame(int32 frame, VectorBase<BaseFloat> *feat) {
      feat->CopyFromVec(frames_info_[frame]->frame);
    }
    

  private:
    
    struct FrameInfo {
      Vector<BaseFloat> frame;
      FrameInfo(int32 dim) { frame.Resize(dim); }
    };

    RawSignalOptions opts_;

    // frame_info_ is indexed by frame-index, from 0 to at most opts_.recompute_frame - 1.
    std::vector<FrameInfo*> frames_info_;

    // This function works out from the signal how many frames are currently available to 
    // process(This is called inside AcceptWaveForm()).
    int32 NumFramesAvailable(int64 num_samples, bool snip_edges) const;
};

/// This function generates raw frame. 
///  Some post-processing techniques can be applied to raw frame extraction.
void ComputeAndProcessRawSignal(const RawSignalOptions &opts,
                                const VectorBase<BaseFloat> &wave,
                                Matrix<BaseFloat> *output); 
} // end of namespace kaldi
