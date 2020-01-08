// chainbin/chain-lattice-to-post.cc

// Copyright      2017  Vimal Manohar

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
#include "hmm/posterior.h"
#include "lat/lattice-functions.h"
#include "chain/chain-den-graph.h"
#include "chain/chain-denominator.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::chain;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Do forward-backward and collect pdf posteriors over a FST.\n"
        "The labels of FST are 1-index pdf labels i.e. pdf-id + 1.\n"
        "The input likelihoods are assumed to be in log-domain.\n"
        "Usage:  chain-fst-to-post [options] <num-pdfs> <fst-in> <likes-rspecifier> "
        "<post-wspecifier>\n"
        "\n";

    BaseFloat fst_scale = 0.0, min_post = 0.01;

    ParseOptions po(usage);
    po.Register("fst-scale", &fst_scale,
                "Scaling factor for the <fst-in> relative to the likelihood");
    po.Register("min-post", &min_post,
                "Minimum posterior to keep");

    ChainTrainingOptions opts;
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string fst_rxfilename,
      likes_rspecifier,
      post_wspecifier;

    int32 num_pdfs;
    ConvertStringToInteger(po.GetArg(1), &num_pdfs);

    fst_rxfilename = po.GetArg(2);
    likes_rspecifier = po.GetArg(3);
    post_wspecifier = po.GetArg(4);

    if (num_pdfs < 0) {
      KALDI_ERR << "num-pdfs is required to be > 0";
    }

    fst::StdVectorFst fst;
    if (!fst_rxfilename.empty()) {
      ReadFstKaldi(fst_rxfilename, &fst);
      KALDI_ASSERT(fst.NumStates() > 0);

      if (fst_scale < 0.0 || fst_scale > 1.0) {
        KALDI_ERR << "Invalid fst-scale; must be in [0.0, 1.0)";
      }

      if (fst_scale != 1.0) {
        fst::ApplyProbabilityScale(fst_scale, &fst);
      }
    }

    fst::RmEpsilon(&fst);

    DenominatorGraph graph(fst, num_pdfs);

    SequentialBaseFloatMatrixReader likes_reader(likes_rspecifier);
    PosteriorWriter posterior_writer(post_wspecifier);

    int32 num_done = 0, num_fail = 0;

    double tot_like = 0.0, num_frames = 0.0;
    for (; !likes_reader.Done(); likes_reader.Next()) {
      std::string key = likes_reader.Key();
      const MatrixBase<BaseFloat> &likes = likes_reader.Value();

      DenominatorComputation computation(opts, graph, 1,
                                         CuMatrix<BaseFloat>(likes));

      int32 num_frames = likes.NumRows();
      BaseFloat log_like = computation.Forward();

      CuMatrix<BaseFloat> cu_post(likes.NumRows(), likes.NumCols());
      Posterior post;

      if (computation.Backward(1.0, &cu_post)) {
        Matrix<BaseFloat> post_mat(likes.NumRows(), likes.NumCols());
        cu_post.CopyToMat(&post_mat);

        post.resize(post_mat.NumRows());
        for (int32 t = 0; t < post_mat.NumRows(); t++) {
          for (int32 d = 0; d < post_mat.NumCols(); d++) {
            BaseFloat weight = post_mat(t, d);
            if (post_mat(t, d) > min_post) {
              post[t].push_back(std::make_pair(d + 1, weight));
            }
          }
        }

        posterior_writer.Write(key, post);
        num_done++;
      } else {
        num_fail++;
        continue;
      }

      tot_like += log_like;
      num_frames += likes.NumRows();
    }

    KALDI_LOG << "Got graph posteriors for " << num_done 
              << " utterances; failed for " << num_fail;
    KALDI_LOG << "Average log-likelihood per frame is " << tot_like / num_frames 
              << " over " << num_frames << " frames";

    return num_done > 0 ? 0 : 1;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
