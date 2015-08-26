// nnet2bin/lattice-nce-test.cc

// Copyright 2014  Vimal Manohar

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
#include "hmm/transition-model.h"
#include "nnet2/nnet-example-functions.h"
#include "nnet2/am-nnet.h"
#include "nnet2/nnet-compute-discriminative-unsupervised.h"
#include "lat/lattice-functions.h"
#include "base/kaldi-math.h"
#include "base/kaldi-types-extra.h"

namespace kaldi {

  void CheckGradients(
                      const TransitionModel &trans_model,
                      const Lattice &lat, 
                      SignedLogDouble nce_old,
                      SignedLogDouble delta,
                      nnet2::NnetDiscriminativeUpdater updater,
                      const Posterior &post) {

    using namespace fst;
    typedef Lattice::Arc Arc;
    typedef Arc::Weight Weight;
    typedef Arc::StateId StateId;
    
    int32 num_pdfs = trans_model.NumPdfs();

    std::vector<int32> state_times;
    int32 max_time = LatticeStateTimes(lat, &state_times);

    for (int32 i = 0; i < 10; i++) {
      int32 t = RandInt(0, max_time - 1);
      int32 j = RandInt(0, num_pdfs);

      Lattice *lat1 = new Lattice(lat);
      
      bool found_gradient = false;

      int32 num_states = lat.NumStates();
      for (StateId s = 0; s < num_states - 1; s++) {
        if (state_times[s] != t) continue;
        for (MutableArcIterator<Lattice> aiter(lat1, s); !aiter.Done(); aiter.Next()) {
          Arc arc(aiter.Value());
          if (trans_model.TransitionIdToPdf(arc.ilabel) == j) {
            double log_p= -arc.weight.Value2();
            arc.weight.SetValue2( -LogAdd(log_p, delta.LogMagnitude()) );
            aiter.SetValue(arc);
            found_gradient = true;
          }
        }
      }

      if (!found_gradient) {
        i--;
        continue;
      }
      found_gradient = false;

      Posterior post1;

      SignedLogDouble nce_new = LatticeForwardBackwardNce(trans_model, *lat1, &post1);

      double gradient;
      Posterior pdf_post;
      ConvertPosteriorToPdfs(trans_model, post, &pdf_post);

      for (int32 l = 0; l < pdf_post[t].size(); l++) {
        if (pdf_post[t][l].first == j) {
          gradient = pdf_post[t][l].second;
          found_gradient = true;
          break;
        }
      }
      KALDI_ASSERT(found_gradient);;

      double gradient_appx = ((nce_new - nce_old) / delta).Value();
      KALDI_LOG 
        << "Computed Gradient is " << gradient
        << "; Actual Gradient is (" << nce_new << " - " << nce_old << ") / " << delta << " = " << gradient_appx << "\n";

      KALDI_ASSERT( kaldi::ApproxEqual( gradient_appx, gradient, 0.1 ) ); 
    }
  }
}


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Test lattice NCE objective and gradient computation.\n"
        "This uses training examples\n"
        "prepared with nnet-get-egs-discriminative-unsupervised\n"
        "\n"
        "Usage:  lattice-nce-test [options] <model-in> <training-examples-in> <model-out>\n"
        "e.g.:\n"
        "lattice-nce-test 1.nnet ark:1.uegs 2.nnet\n";
    
    bool binary_write = true;
    std::string use_gpu = "yes";
    double delta = 1.0e-10;

    NnetDiscriminativeUpdateOptions update_opts;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");
    po.Register("delta", &delta, "Delta for approximating NCE gradient");
    update_opts.Register(&po);

    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);

    int64 num_examples = 0;

    TransitionModel trans_model;
    {
      AmNnet am_nnet;
      {
        bool binary_read;
        Input ki(nnet_rxfilename, &binary_read);
        trans_model.Read(ki.Stream(), binary_read);
        am_nnet.Read(ki.Stream(), binary_read);
      }

      NnetDiscriminativeStats stats(trans_model.NumPdfs());
      SequentialDiscriminativeNnetExampleReader example_reader(examples_rspecifier);

      for (; !example_reader.Done(); example_reader.Next(), num_examples++) {
        NnetDiscriminativeUpdater updater (am_nnet, trans_model, update_opts,
                                 example_reader.Value(), NULL, &stats);
        updater.Update();
        {
          Lattice lat1(updater.GetLattice());

          Posterior post1;
          SignedLogDouble nce1 = LatticeForwardBackwardNce(trans_model, lat1, &post1);

          CheckGradients(trans_model, lat1, nce1, SignedLogDouble(static_cast<double>(delta)), updater, post1);

        }

        if (GetVerboseLevel() >= 3) 
          stats.Print("nce");
        else {
          if (num_examples % 10 == 0 && num_examples != 0) { // each example might be 500 frames.
            if (GetVerboseLevel() >= 2) {
              stats.Print("nce");
            }
          }          
        }
      }

      stats.Print("nce");
        
      {
        Output ko(nnet_wxfilename, binary_write);
        trans_model.Write(ko.Stream(), binary_write);
        am_nnet.Write(ko.Stream(), binary_write);
      }
    }
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    KALDI_LOG << "Finished training, processed " << num_examples
              << " training examples.  Wrote model to "
              << nnet_wxfilename;
    return (num_examples == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



