// latbin/lattice-determinize-phone-pruned-parallel.cc

// Copyright 2014  Guoguo Chen

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
#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/lattice-functions.h"
#include "lat/push-lattice.h"
#include "util/common-utils.h"
#include "util/kaldi-thread.h"

namespace kaldi {

class DeterminizeLatticeTask {
 public:
  // Initializer takes ownership of "lat".
  DeterminizeLatticeTask(
      const TransitionModel &trans_model,
      fst::DeterminizeLatticePhonePrunedOptions &opts,
      std::string key,
      BaseFloat acoustic_scale,
      BaseFloat beam,
      Lattice *lat,
      CompactLatticeWriter *clat_writer,
      LatticeWriter *lat_writer,
      int32 *num_warn):
      trans_model_(&trans_model), opts_(opts), key_(key),
      acoustic_scale_(acoustic_scale), beam_(beam),
      lat_(lat), clat_writer_(clat_writer), 
      lat_writer_(lat_writer), num_warn_(num_warn) { 
        KALDI_ASSERT((lat_writer_ && !clat_writer_) || 
                     (!lat_writer_ && clat_writer_)); 
      }

  void operator () () {
    if (lat_writer_)
      ComputeAcousticScoresMap(*lat_, &acoustic_scores_);

    // We apply the acoustic scale before determinization and will undo it
    // afterward, since it can affect the result.
    fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale_), lat_);

    if (!DeterminizeLatticePhonePrunedWrapper(
            *trans_model_, lat_, beam_, &det_clat_, opts_)) {
      KALDI_WARN << "For key " << key_ << ", determinization did not succeed"
          "(partial output will be pruned tighter than the specified beam.)";
      (*num_warn_)++;
    }

    delete lat_;
    lat_ = NULL;
  }

  ~DeterminizeLatticeTask() {
    if (clat_writer_) {
      KALDI_VLOG(2) << "Wrote lattice with " << det_clat_.NumStates()
                    << " for key " << key_;
      // Invert the original acoustic scaling
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0/acoustic_scale_),
                        &det_clat_);
      clat_writer_->Write(key_, det_clat_);
    } else {
      KALDI_VLOG(2) << "Wrote lattice with " << det_clat_.NumStates()
                    << " for key " << key_;
      Lattice out_lat;
      fst::ConvertLattice(det_clat_, &out_lat);
        
      // Replace each arc (t, tid) with the averaged acoustic score from
      // the computed map
      ReplaceAcousticScoresFromMap(acoustic_scores_, &out_lat);

      lat_writer_->Write(key_, out_lat);
    }
  }
 private:
  const TransitionModel *trans_model_;
  const fst::DeterminizeLatticePhonePrunedOptions &opts_;
  std::string key_;
  BaseFloat acoustic_scale_;
  BaseFloat beam_;
  // The lattice we're working on. Owned locally.
  Lattice *lat_;
  // The output of our process. Will be written to clat_writer_ in the
  // destructor.
  CompactLattice det_clat_;
  CompactLatticeWriter *clat_writer_;
  LatticeWriter *lat_writer_;
  int32 *num_warn_;

  // Used to compute a map from each (t, tid) to (sum_of_acoustic_scores, count)
  unordered_map<std::pair<int32,int32>, std::pair<BaseFloat, int32>,
                                      PairHasher<int32> > acoustic_scores_;
};

} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    
    const char *usage =
        "Determinize lattices, keeping only the best path (sequence of\n"
        "acoustic states) for each input-symbol sequence. This is a version\n"
        "of lattice-determinize-phone-pruned that accepts the --num-threads\n"
        "option. The program does phone insertion when doing a first pass\n"
        "determinization, it then removes the inserted symbols and does a\n"
        "second pass determinization. It also does pruning as part of the\n"
        "determinization algorithm, which is more efficient and prevents\n"
        "blowup.\n"
        "\n"
        "Usage: lattice-determinize-phone-pruned-parallel [options] \\\n"
        "                 <model> <lattice-rspecifier> <lattice-wspecifier>\n"
        " e.g.: lattice-determinize-phone-pruned-parallel \\\n"
        "           --acoustic-scale=0.1 final.mdl ark:in.lats ark:det.lats\n";
    
    ParseOptions po(usage);
    bool write_compact = true;
    BaseFloat acoustic_scale = 1.0;
    BaseFloat beam = 10.0;

    TaskSequencerConfig sequencer_opts;
    fst::DeterminizeLatticePhonePrunedOptions determinize_opts;
    determinize_opts.max_mem = 50000000;
    
    po.Register("write-compact", &write_compact, 
                "If true, write in normal (compact) form. "
                "--write-compact=false allows you to retain frame-level "
                "acoustic score information, but this requires the input "
                "to be in non-compact form e.g. undeterminized lattice "
                "straight from decoding.");
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic"
                " likelihoods.");
    po.Register("beam", &beam, "Pruning beam [applied after acoustic scaling].");
    determinize_opts.Register(&po);
    sequencer_opts.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        lats_rspecifier = po.GetArg(2),
        lats_wspecifier = po.GetArg(3);

    TransitionModel trans_model;
    ReadKaldiObject(model_rxfilename, &trans_model);

    // Reads as regular lattice-- this is the form the determinization code
    // accepts.
    SequentialLatticeReader lat_reader(lats_rspecifier);
    
    CompactLatticeWriter *compact_lat_writer = NULL;
    LatticeWriter *lat_writer = NULL;

    if (write_compact)
      compact_lat_writer = new CompactLatticeWriter(lats_wspecifier);
    else
      lat_writer = new LatticeWriter(lats_wspecifier);

    TaskSequencer<DeterminizeLatticeTask> sequencer(sequencer_opts);

    int32 n_done = 0, n_warn = 0;

    if (acoustic_scale == 0.0)
      KALDI_ERR << "Do not use a zero acoustic scale (cannot be inverted)";

    for (; !lat_reader.Done(); lat_reader.Next()) {
      std::string key = lat_reader.Key();

      // Will give ownership to "task" below.
      Lattice *lat = lat_reader.Value().Copy();

      KALDI_VLOG(2) << "Processing lattice " << key;

      DeterminizeLatticeTask *task = new DeterminizeLatticeTask(
          trans_model, determinize_opts, key, acoustic_scale, beam,
          lat, compact_lat_writer, lat_writer, &n_warn);
      sequencer.Run(task);

      n_done++;
    }
    sequencer.Wait();
    KALDI_LOG << "Done " << n_done << " lattices, determinization finished "
              << "earlier than specified by the beam on " << n_warn << " of "
              << "these.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
