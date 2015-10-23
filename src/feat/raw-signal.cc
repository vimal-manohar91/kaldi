// feat/raw-signal.cc

// Copyright 2015 Pegah Ghahremani

#include "raw-signal.h"



namespace kaldi {
int32 NumRawFrames(int32 nsamp,
                   const RawSignalOptions &opts) {
  
  int32 frame_length = opts.WindowSize();
  KALDI_ASSERT(frame_length != 0);
  if (opts.snip_edges) {
    if (static_cast<int32>(nsamp) < frame_length) {
      return 0;
    } else {
      return ((nsamp / frame_length));
    }
  } else {
    return (int32)(nsamp * 1.0f / frame_length + 0.5f);
    // if snip-edges=false, the number of frames would be determined by 
    // rounding the (file-length / frame_length) to the nearest integer.
  }
}  
 
void ComputeRawFrames::ExtractRawFrames(const VectorBase<BaseFloat> &wave,
                                       int32 f, // frame-number
                                       VectorBase<BaseFloat> *window) const {

  int32 frame_length = opts_.WindowSize();
  KALDI_ASSERT(frame_length != 0);
  Vector<BaseFloat> wave_part(frame_length);
  if (opts_.snip_edges) {
    int32 start=frame_length * f, end= start + frame_length;
    KALDI_ASSERT(start >= 0 && end <= wave.Dim());  
    wave_part.CopyFromVec(wave.Range(start, frame_length)); 
  } else {
    // If opts.snip_edges = false, we allow the frames to go slightly over the
    // edges of the file; we'll extend the data by reflection.
    int32 mid = frame_length * (f + 0.5),
        begin = mid - frame_length / 2,
        end = begin + frame_length,
        begin_limited = std::max<int32>(0, begin),
        end_limited = std::min(end, wave.Dim()),
        length_limited = end_limited - begin_limited;

    // Copy the main part.  Usually this will be the entire window.
    wave_part.Range(begin_limited - begin, length_limited).
        CopyFromVec(wave.Range(begin_limited, length_limited));
    
    // Deal with any end effects by reflection, if needed.  This code will
    // rarely be reached, so we don't concern ourselves with efficiency.
    for (int32 f = begin; f < 0; f++) {
      int32 reflected_f = -f;
      // The next statement will only have an effect in the case of files
      // shorter than a single frame, it's to avoid a crash in those cases.
      reflected_f = reflected_f % wave.Dim(); 
      wave_part(f - begin) = wave(reflected_f);
    }
    for (int32 f = wave.Dim(); f < end; f++) {
      int32 distance_to_end = f - wave.Dim();
      // The next statement will only have an effect in the case of files
      // shorter than a single frame, it's to avoid a crash in those cases.
      distance_to_end = distance_to_end % wave.Dim();
      int32 reflected_f = wave.Dim() - 1 - distance_to_end;
      wave_part(f - begin) = wave(reflected_f);
    }
  }
  SubVector<BaseFloat> window_part(*window, 0, frame_length);
  window->CopyFromVec(wave_part);
/*
if (opts.dither != 0.0) Dither(window, opts.dither);

if (opts.remove_dc_offset)
  window->Add(-window->Sum() / frame_length);

if (log_energy_pre_window != NULL) {
  BaseFloat energy = std::max(VecVec(window_part, window_part),
                              std::numeric_limits<BaseFloat>::min());
  *log_energy_pre_window = Log(energy);
}

if (opts.preemph_coeff != 0.0)
  Preemphasize(&window_part, opts.preemph_coeff);
SubVector<BaseFloat>(*window, frame_length, frame_length
}
*/
}

void ComputeRawFrames::AcceptWaveForm(const VectorBase<BaseFloat> &wave) {
  int32 end_frame = NumRawFrames(wave.Dim(), opts_);

  int32 full_frame_length = opts_.WindowSize(),
  start_frame = frames_info_.size(),
    num_new_frames = end_frame - start_frame;

  Vector<BaseFloat> window(full_frame_length);
  for (int32 frame = start_frame; frame < end_frame; frame++) {
    ExtractRawFrames(wave, frame, &window);
    FrameInfo *frame_info = new FrameInfo(full_frame_length);
    frame_info->frame.CopyFromVec(window);
    frames_info_.push_back(frame_info);
  }
}

void ComputeAndProcessRawSignal(const RawSignalOptions &opts,
                           const VectorBase<BaseFloat> &wave,
                           Matrix<BaseFloat> *output) {

  int32 cur_rows = 100;

  ComputeRawFrames frame_extractor(opts);

  int32 cur_offset = 0, cur_frame = 0, 
    samp_per_chunk = 0; // opts.frames_per_chunk * opts.samp_freq * 1.0e-03 * opts.frame_length_ms;
                        // It can be set in future by adding opts.frames_per_chunk
                        // for online feature extraction
  Matrix<BaseFloat> feats(cur_rows, frame_extractor.Dim());

  while (cur_offset < wave.Dim()) {
    int32 num_samp;

    if (samp_per_chunk > 0)
      num_samp = std::min(samp_per_chunk, wave.Dim() - cur_offset);
    else // user 
      num_samp = wave.Dim();
    SubVector<BaseFloat> wave_chunk(wave, cur_offset, num_samp);
    frame_extractor.AcceptWaveForm(wave_chunk);
    cur_offset += num_samp;

    //if (cur_offset == wave.Dim())
    //  frame_extractor.InputFinished();
    

    // Get each frame as soon as it is ready
    for (; cur_frame < frame_extractor.NumFramesReady(); cur_frame++) {
      if (cur_frame >= cur_rows) {
        cur_rows *= 2;
        feats.Resize(cur_rows, frame_extractor.Dim(), kCopyData);

      }
      SubVector<BaseFloat> row(feats, cur_frame);
      frame_extractor.GetFrame(cur_frame, &row);
    }
    output->Resize(cur_frame-1, frame_extractor.Dim());
    output->CopyFromMat(feats.RowRange(0, cur_frame-1));
  }
}

} // end of namespace kaldi
