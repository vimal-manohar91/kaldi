// featbin/compute-zero-crossings.cc

// Copyright 2015   Vimal Manohar

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
#include "feat/wave-reader.h"
#include "feat/feature-window.h"

namespace kaldi {

void ComputeZeroCrossings(const VectorBase<BaseFloat> &wave,
                          FrameExtractionOptions frame_opts, 
                          BaseFloat threshold, 
                          Vector<BaseFloat> *output,
                          Vector<BaseFloat> *wave_remainder) {
  KALDI_ASSERT(output != NULL);

  // Get dimensions of output features
  int32 rows_out = NumFrames(wave.Dim(), frame_opts);
  if (rows_out == 0)
    KALDI_ERR << "No frames fit in file (#samples is " << wave.Dim() << ")";
  // Prepare the output buffer
  output->Resize(rows_out);
  
  // Optionally extract the remainder for further processing
  if (wave_remainder != NULL)
    ExtractWaveformRemainder(wave, frame_opts, wave_remainder);

  // Buffers
  Vector<BaseFloat> window;  // windowed waveform.
  
  FeatureWindowFunction feature_window_function(frame_opts);
  // Compute all the freames, r is frame index..
  for (int32 r = 0; r < rows_out; r++) {
    // Cut the window, apply window function
    ExtractWindow(0, wave, r, frame_opts, feature_window_function,
                  &window, NULL);
    
    int32 zc = 0;
    for (int32 i = 1; i < window.Dim(); i++) {
      if ( (window(i-1) < -threshold && window(i) > threshold) 
          || (window(i-1) > threshold && window(i) < -threshold) ) {
        zc++;
      }
    }

    (*output)(r) = zc;
  }
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Create zero-crossing features\n"
        "Usage:  compute-zero-crossings [options...] <wav-rspecifier> <feats-wspecifier>\n";

    // construct all the global objects
    ParseOptions po(usage);
    FrameExtractionOptions opts;
    
    int32 channel = -1;
    BaseFloat min_duration = 0.0, zero_crossing_threshold = 0.0;
    bool write_as_vector = false;

    // Register the option struct
    opts.Register(&po);
    // Register the options
    po.Register("channel", &channel, "Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right)");
    po.Register("min-duration", &min_duration, "Minimum duration of segments to process (in seconds).");
    po.Register("zero-crossing-threshold", &zero_crossing_threshold, 
                "Take any value within this threshold as zero "
                "for zero crossing computation");
    po.Register("write-as-vector", &write_as_vector, "Write as a vector "
                "to interpret the output as weights instead of "
                "a column of feature matrix");

    // OPTION PARSING ..........................................................
    //

    // parse options (+filling the registered variables)
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1);
    std::string output_wspecifier = po.GetArg(2);

    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    BaseFloatMatrixWriter *matrix_writer = NULL;
    BaseFloatVectorWriter *vector_writer = NULL;

    if (write_as_vector)
      vector_writer = new BaseFloatVectorWriter(output_wspecifier);
    else 
      matrix_writer = new BaseFloatMatrixWriter(output_wspecifier);

    int32 num_utts = 0, num_success = 0;
    for (; !reader.Done(); reader.Next()) {
      num_utts++;
      std::string utt = reader.Key();
      const WaveData &wave_data = reader.Value();
      if (wave_data.Duration() < min_duration) {
        KALDI_WARN << "File: " << utt << " is too short ("
                   << wave_data.Duration() << " sec): producing no output.";
        continue;
      }
      int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
      {  // This block works out the channel (0=left, 1=right...)
        KALDI_ASSERT(num_chan > 0);  // should have been caught in
        // reading code if no channels.
        if (channel == -1) {
          this_chan = 0;
          if (num_chan != 1)
            KALDI_WARN << "Channel not specified but you have data with "
                       << num_chan  << " channels; defaulting to zero";
        } else {
          if (this_chan >= num_chan) {
            KALDI_WARN << "File with id " << utt << " has "
                       << num_chan << " channels but you specified channel "
                       << channel << ", producing no output.";
            continue;
          }
        }
      }

      if (opts.samp_freq != wave_data.SampFreq())
        KALDI_ERR << "Sample frequency mismatch: you specified "
                  << opts.samp_freq << " but data has "
                  << wave_data.SampFreq() << " (use --sample-frequency "
                  << "option).  Utterance is " << utt;

      SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
      Vector<BaseFloat> zero_crossings;
      ComputeZeroCrossings(waveform, opts, zero_crossing_threshold, &zero_crossings, NULL);

      if (write_as_vector) {
        vector_writer->Write(utt, zero_crossings);
      } else {
        Matrix<BaseFloat> mat(zero_crossings.Dim(), 1);
        mat.CopyColFromVec(zero_crossings, 0); 
        matrix_writer->Write(utt, mat);
      }
      
      if(num_utts % 10 == 0)
        KALDI_LOG << "Processed " << num_utts << " utterances";
      KALDI_VLOG(2) << "Processed features for key " << utt;
      num_success++;
    }
    KALDI_LOG << " Done " << num_success << " out of " << num_utts
              << " utterances.";

    if (vector_writer != NULL) delete vector_writer;
    if (matrix_writer != NULL) delete matrix_writer;
    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

