// feat/raw-signal-test.cc

// Copyright 2015 Pegah Ghahremani

#include <iostream>
#include "base/kaldi-math.h"
#include "base/timer.h"
#include "feat/wave-reader.h"    
#include "sys/stat.h" 
#include "sys/types.h"
#include "feat/raw-signal.h" 

namespace kaldi {

static void UnitTestSimple() {
  KALDI_LOG << "=== UnitTestSimple() ===";
  std::ifstream is("test_data/test.wav"); 

  WaveData wave; 
  wave.Read(is);
  const Matrix<BaseFloat> data(wave.Data());
  KALDI_ASSERT(data.NumRows() == 1);
  Vector<BaseFloat> waveform(data.Row(0));

  RawSignalOptions raw_opts;

  // Compute raw frames

  Matrix<BaseFloat> raw_feats;
  ComputeAndProcessRawSignal(raw_opts, waveform, &raw_feats);
  std::string raw_str = "test_data/raw-feats.txt";
  std::ofstream os_raw_feats(raw_str.c_str());
  raw_feats.Write(os_raw_feats, false);
}
static void UnitTestFeat() {
  UnitTestSimple();
}

} // end of namespace kaldi

int main() {
  using namespace kaldi;             

  try {
    for (int i = 0; i < 1; i++)
      UnitTestFeat();
    std::cout << "Tests succeeded.\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return 1;
  }
}


