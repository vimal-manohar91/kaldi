// chainbin/nnet3-chain-get-egs.cc

// Copyright      2015  Johns Hopkins University (author:  Daniel Povey)

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

#include <sstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-chain-example.h"
#include "nnet3/nnet-example-utils.h"
#include "lat/lattice-functions.h"
#include "chain/chain-supervision.h"

namespace kaldi {
namespace nnet3 {

/** This function converts lattice to fst with weight equal to
    sum of acoustic and language score.
*/
void ConvertLatticeToPdfLabels(
    const TransitionModel &tmodel,
    const Lattice &ifst,
    fst::StdVectorFst *ofst) {
  typedef fst::ArcTpl<LatticeWeight> ArcIn;
  typedef fst::StdArc ArcOut;
  typedef ArcIn::StateId StateId;
  ofst->DeleteStates();
  // The states will be numbered exactly the same as the original FST.
  // Add the states to the new FST.
  StateId num_states = ifst.NumStates();
  for (StateId s = 0; s < num_states; s++)
    ofst->AddState();
  ofst->SetStart(ifst.Start());
  for (StateId s = 0; s < num_states; s++) {
    LatticeWeight final_iweight = ifst.Final(s);
    if (final_iweight != LatticeWeight::Zero()) {
      fst::TropicalWeight final_oweight;
      ConvertLatticeWeight(final_iweight, &final_oweight);
      ofst->SetFinal(s, final_oweight);
    }
    for (fst::ArcIterator<Lattice> iter(ifst, s);
         !iter.Done();
         iter.Next()) {
      ArcIn arc = iter.Value();
      KALDI_PARANOID_ASSERT(arc.weight != LatticeWeight::Zero());
      ArcOut oarc;
      ConvertLatticeWeight(arc.weight, &oarc.weight);
      oarc.ilabel = tmodel.TransitionIdToPdf(arc.ilabel) + 1;
      oarc.olabel = tmodel.TransitionIdToPdf(arc.ilabel) + 1;
      oarc.nextstate = arc.nextstate;
      ofst->AddArc(s, oarc);
    }
  }
}


/**
   This function does all the processing for one utterance, and outputs the
   supervision objects to 'example_writer'.  Note: if normalization_fst is the
   empty FST (with no states), it skips the final stage of egs preparation and
   you should do it later with nnet3-chain-normalize-egs.
*/

static bool ProcessFile(const GeneralMatrix &feats,
                        const MatrixBase<BaseFloat> *ivector_feats,
                        int32 ivector_period,
                        const Posterior &pdf_post,
                        BaseFloat min_post,
                        const std::string &utt_id,
                        bool compress,
                        int32 num_pdfs,
                        UtteranceSplitter *utt_splitter,
                        NnetExampleWriter *example_writer) {
  //KALDI_ASSERT(supervision.num_sequences == 1);
  int32 num_input_frames = feats.NumRows();
  int32 num_output_frames = pdf_post.size();

  if (!utt_splitter->LengthsMatch(utt_id, num_input_frames, num_output_frames))
    return false;  // LengthsMatch() will have printed a warning.

  std::vector<ChunkTimeInfo> chunks;

  utt_splitter->GetChunksForUtterance(num_input_frames, &chunks);

  if (chunks.empty()) {
    KALDI_WARN << "Not producing egs for utterance " << utt_id
               << " because it is too short: "
               << num_input_frames << " frames.";
    return false;
  }

  int32 frame_subsampling_factor = utt_splitter->Config().frame_subsampling_factor;

  for (size_t c = 0; c < chunks.size(); c++) {
    ChunkTimeInfo &chunk = chunks[c];

    int32 tot_input_frames = chunk.left_context + chunk.num_frames +
        chunk.right_context,
        start_frame = chunk.first_frame - chunk.left_context;

    GeneralMatrix input_frames;
    ExtractRowRangeWithPadding(feats, start_frame, tot_input_frames,
                               &input_frames);

    NnetExample eg;
    // call the regular input "input".
    eg.io.push_back(NnetIo("input", -chunk.left_context, input_frames));

    if (ivector_feats != NULL) {
      // if applicable, add the iVector feature.
      // choose iVector from a random frame in the chunk
      int32 ivector_frame = RandInt(start_frame,
                                    start_frame + num_input_frames - 1),
          ivector_frame_subsampled = ivector_frame / ivector_period;
      if (ivector_frame_subsampled < 0)
        ivector_frame_subsampled = 0;
      if (ivector_frame_subsampled >= ivector_feats->NumRows())
        ivector_frame_subsampled = ivector_feats->NumRows() - 1;
      Matrix<BaseFloat> ivector(1, ivector_feats->NumCols());
      ivector.Row(0).CopyFromVec(ivector_feats->Row(ivector_frame_subsampled));
      eg.io.push_back(NnetIo("ivector", 0, ivector));
    }

    // Note: chunk.first_frame and chunk.num_frames will both be
    // multiples of frame_subsampling_factor.
    int32 start_frame_subsampled = chunk.first_frame / frame_subsampling_factor,
        num_frames_subsampled = chunk.num_frames / frame_subsampling_factor;

    // Substract 1 from post to convert it back to pdf-id.
    Posterior labels(num_frames_subsampled);

    for (int i = 0; i < num_frames_subsampled; i++) {
      int t = i + start_frame_subsampled;
      if (t < pdf_post.size()) {
        for (int32 j = 0; j < pdf_post[t].size(); j++) {
          BaseFloat post = pdf_post[t][j].second;
          if (post > min_post) {
            labels[i].push_back(std::make_pair(
                  pdf_post[t][j].first - 1, post));
          }
        }
      }
      for (std::vector<std::pair<int32, BaseFloat> >::iterator
              iter = labels[i].begin(); iter != labels[i].end(); ++iter)
        iter->second *= chunk.output_weights[i];
    }

    SubVector<BaseFloat> output_weights(
        &(chunk.output_weights[0]),
        static_cast<int32>(chunk.output_weights.size()));

    eg.io.push_back(NnetIo("output", num_pdfs, 0, labels));

    if (compress)
      eg.Compress();

    std::ostringstream os;
    os << utt_id << "-" << chunk.first_frame;

    std::string key = os.str(); // key is <utt_id>-<frame_id>

    example_writer->Write(key, eg);
  }
  return true;
}

} // namespace nnet3
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    using namespace kaldi::chain;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get frame-by-frame examples of data for nnet3+chain neural network\n"
        "training.  This involves breaking up utterances into pieces of a\n"
        "fixed size. \n"
        "The input is lattice and it will transform into new lattice "
        "with pdf labels. The it will compose with <normalization-fst> "
        "and does forward backward to get posterior.\n"
        "This egs generation can be used for teacher student learning setup \n"
        "where the lattice extracted from teacher network.\n"
        "Note: if <normalization-fst> is not supplied the egs will not be\n"
        "ready for training; in that case they should later be processed\n"
        "with nnet3-chain-normalize-egs\n"
        "\n"
        "Usage:  nnet3-chain-get-egs-post [options] [<normalization-fst>] <features-rspecifier> "
        "<lattice-rspecifier> <egs-wspecifier>\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "nnet3-chain-get-egs-post --left-context=25 --right-context=9\n"
        "--num-frames=20 dir/normalization.fst \"$feats\" \n"
        "ark:lat.1.ark ark:cegs.1.ark";

    bool compress = true;
    int32 length_tolerance = 100, online_ivector_period = 1;

    ExampleGenerationConfig eg_config;  // controls num-frames,
                                        // left/right-context, etc.

    int32 srand_seed = 0;
    std::string online_ivector_rspecifier,
      deriv_weights_rspecifier;
    BaseFloat min_post = 1e-8, normalization_scale = 0.5;

    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs with input features "
                "in compressed format (recommended).  Update: this is now "
                "only relevant if the features being read are un-compressed; "
                "if already compressed, we keep we same compressed format when "
                "dumping-egs.");
    po.Register("ivectors", &online_ivector_rspecifier, "Alias for "
                "--online-ivectors option, for back compatibility");
    po.Register("online-ivectors", &online_ivector_rspecifier, "Rspecifier of "
                "ivector features, as a matrix."
                "--online-ivectors option");
    po.Register("online-ivector-period", &online_ivector_period, "Number of "
                "frames between iVectors in matrices supplied to the "
                "--online-ivectors option");
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    po.Register("length-tolerance", &length_tolerance, "Tolerance for "
                "difference in num-frames between feat and ivector matrices");
    po.Register("min-post", &min_post, "Minimum posterior to keep; this will "
                "avoid dumping out all posteriors.");
    po.Register("normalization-scale", &normalization_scale,
                "Scale normalization FST");
    po.Register("deriv-weights-rspecifier", &deriv_weights_rspecifier,
                "Not implemented");

    eg_config.Register(&po);

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() < 4 || po.NumArgs() > 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        normalization_fst_rxfilename,
        trans_model,
        feature_rspecifier,
        lattice_rspecifier,
        examples_wspecifier;
    if (po.NumArgs() == 4) {
      trans_model = po.GetArg(1);
      feature_rspecifier = po.GetArg(2);
      lattice_rspecifier = po.GetArg(3);
      examples_wspecifier = po.GetArg(4);
    } else {
      normalization_fst_rxfilename = po.GetArg(1);
      KALDI_ASSERT(!normalization_fst_rxfilename.empty());
      trans_model = po.GetArg(2);
      feature_rspecifier = po.GetArg(3);
      lattice_rspecifier = po.GetArg(4);
      examples_wspecifier = po.GetArg(5);
    }

    TransitionModel tmodel;
    ReadKaldiObject(trans_model, &tmodel);

    eg_config.ComputeDerived();
    UtteranceSplitter utt_splitter(eg_config);

    fst::StdVectorFst normalization_fst;
    if (!normalization_fst_rxfilename.empty()) {
      ReadFstKaldi(normalization_fst_rxfilename, &normalization_fst);
      KALDI_ASSERT(normalization_fst.NumStates() > 0);

      ApplyProbabilityScale(0.5, &normalization_fst); 
    }

    // Read as GeneralMatrix so we don't need to un-compress and re-compress
    // when selecting parts of matrices.
    SequentialGeneralMatrixReader feat_reader(feature_rspecifier);
    //chain::RandomAccessSupervisionReader supervision_reader(
    //    supervision_rspecifier);
    RandomAccessLatticeReader lattice_reader(lattice_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);
    RandomAccessBaseFloatMatrixReader online_ivector_reader(
        online_ivector_rspecifier);

    int32 num_err = 0;

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const GeneralMatrix &feats = feat_reader.Value();
      if (!lattice_reader.HasKey(key)) {
        KALDI_WARN << "No pdf-level posterior for key " << key;
        num_err++;
      } else {
        //const chain::Supervision &supervision = supervision_reader.Value(key);
        Lattice lat = lattice_reader.Value(key);
        const Matrix<BaseFloat> *online_ivector_feats = NULL;
        if (!online_ivector_rspecifier.empty()) {
          if (!online_ivector_reader.HasKey(key)) {
            KALDI_WARN << "No iVectors for utterance " << key;
            num_err++;
            continue;
          } else {
            // this address will be valid until we call HasKey() or Value()
            // again.
            online_ivector_feats = &(online_ivector_reader.Value(key));
          }
        }
        if (online_ivector_feats != NULL &&
            (abs(feats.NumRows() - (online_ivector_feats->NumRows() *
                                    online_ivector_period)) > length_tolerance
             || online_ivector_feats->NumRows() == 0)) {
          KALDI_WARN << "Length difference between feats " << feats.NumRows()
                     << " and iVectors " << online_ivector_feats->NumRows()
                     << "exceeds tolerance " << length_tolerance;
          num_err++;
          continue;
        }

        fst::StdVectorFst sup_fst;
        fst::ScaleLattice(fst::GraphLatticeScale(0.5), &lat);
        ConvertLatticeToPdfLabels(tmodel, lat, &sup_fst);

        if (normalization_fst.NumStates() > 0 &&
            !chain::AddWeightToFst(normalization_fst, &sup_fst)) {
          KALDI_WARN << "For utterance " << key << ", feature frames "
                     << ", FST was empty after composing with normalization FST. "
                     << "This should be extremely rare (a few per corpus, at most)";
        }

        // Convert fst to lattice to extract posterior using forward backward.
        Lattice sup_lat;
        ConvertFstToLattice(sup_fst, &sup_lat);
        Posterior pdf_post;
        LatticeForwardBackward(sup_lat, &pdf_post);

        if (!ProcessFile(feats, online_ivector_feats, online_ivector_period,
                         pdf_post, min_post, key, compress, tmodel.NumPdfs(),
                         &utt_splitter, &example_writer))
          num_err++;
      }
    }
    if (num_err > 0)
      KALDI_WARN << num_err << " utterances had errors and could "
          "not be processed.";
    // utt_splitter prints stats in its destructor.
    return utt_splitter.ExitStatus();
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
