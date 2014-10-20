// bin/prob-to-max-post.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "hmm/posterior.h"

/* Convert a matrix probabilities 
   to something of type Posterior, i.e. for each utterance, a
   vector<vector<pair<int32, BaseFloat> > >, which is a sparse representation
   of the probabilities.
   To avoid getting very tiny values making it non-sparse, we support
   thresholding, and this can either be done as a simple threshold, or (the
   default) a pseudo-random thing where you preserve the expectation, e.g.
   if the threshold is 0.01 and the value is 0.001, it will be zero with
   probability 0.9 and 0.01 with probability 0.1.
*/

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Convert a matrix of probabilities (e.g. from nnet-prob) to posteriors\n"
        "Usage:  prob-to-max-post [options] <prob-matrix-rspecifier> <posteriors-wspecifier>\n"
        "e.g.:\n"
        " nnet-prob [args] | prob-to-max-post ark:- ark:1.post\n"
        "Caution: in this particular example, the output would be posteriors of pdf-ids,\n"
        "rather than transition-ids (c.f. post-to-pdf-post)\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string prob_rspecifier = po.GetArg(1);
    std::string posteriors_wspecifier = po.GetArg(2);

    int32 num_done = 0;
    SequentialBaseFloatMatrixReader prob_reader(prob_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);

    for (; !prob_reader.Done(); prob_reader.Next()) {
      num_done++;
      const Matrix<BaseFloat> &probs = prob_reader.Value();
      // Posterior is vector<vector<pair<int32, BaseFloat> > >
      Posterior post(probs.NumRows());
      for (int32 i = 0; i < probs.NumRows(); i++) {
        SubVector<BaseFloat> row(probs, i);
        BaseFloat p = row(0);
        BaseFloat max_post = p;
        post[i].push_back(std::make_pair(0, p));
        for (int32 j = 1; j < row.Dim(); j++) {
          p = row(j);
          if (p > max_post) {
            post[i][0].first = j;
            post[i][0].second = p;
            max_post = p;
          }
        }
      }
      posterior_writer.Write(prob_reader.Key(), post);
    }
    KALDI_LOG << "Converted " << num_done << " prob matrices to posteriors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



