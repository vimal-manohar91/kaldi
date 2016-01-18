// nnet3bin/nnet3-discriminative-train.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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
#include "nnet3/nnet-discriminative-training.h"
#include "nnet3/nnet-discriminative-diagnostics.h"
#include "nnet3/am-nnet-simple.h"
#include "lat/lattice-functions.h"

namespace kaldi {
namespace nnet3 {

void UnitTestMergeExamples(const TransitionModel &tmodel,
                           const std::vector<int32> &silence_phones,
                           const std::string criterion,
                           bool one_silence_class,
                           std::vector<NnetDiscriminativeExample> &examples, 
                           NnetDiscriminativeComputeObjf *computer) {
  NnetDiscriminativeExample merged_eg;
  MergeDiscriminativeExamples(false, &examples, &merged_eg);

  Posterior post_merged;
  Posterior post_smbr_merged;

  std::vector<discriminative::DiscriminativeTrainingStats> stats_list;

  double objf = 0.0;

  for (size_t i = 0; i < examples.size(); i++) {
    Posterior post;
    LatticeForwardBackward(examples[i].outputs[0].supervision.den_lat, &post, NULL);
    Posterior post_smbr;
    //LatticeForwardBackwardMpeVariants(tmodel, silence_phones, examples[i].outputs[0].supervision.den_lat, examples[i].outputs[0].supervision.num_ali, criterion, one_silence_class, &post_smbr);

    post_merged.insert(post_merged.end(), post.begin(), post.end());
    post_smbr_merged.insert(post_smbr_merged.end(), post_smbr.begin(), post_smbr.end());

    computer->Compute(examples[i]);
    const discriminative::DiscriminativeTrainingStats &stats = computer->Stats();
    stats_list.push_back(stats);
    objf += stats.TotalObjf(criterion);
    computer->Reset();
  }

  Posterior merged_post;
  LatticeForwardBackward(merged_eg.outputs[0].supervision.den_lat, &merged_post, NULL);
  Posterior merged_post_smbr;
  //LatticeForwardBackwardMpeVariants(tmodel, silence_phones, merged_eg.outputs[0].supervision.den_lat, merged_eg.outputs[0].supervision.num_ali, criterion, one_silence_class, &merged_post_smbr);

  computer->Compute(merged_eg);
  const discriminative::DiscriminativeTrainingStats &merged_stats = computer->Stats();
  
  KALDI_ASSERT(merged_post_smbr.size() == post_smbr_merged.size());
  for (size_t i = 0; i < post_smbr_merged.size(); i++) {
    for (size_t j = 0; j < post_smbr_merged[i].size(); j++) {
      KALDI_ASSERT(post_smbr_merged[i][j].first == merged_post_smbr[i][j].first);
      KALDI_ASSERT(kaldi::ApproxEqual(post_smbr_merged[i][j].second, merged_post_smbr[i][j].second));
    }
  }

  KALDI_ASSERT(merged_post.size() == post_merged.size());
  for (size_t i = 0; i < post_merged.size(); i++) {
    for (size_t j = 0; j < post_merged[i].size(); j++) {
      KALDI_ASSERT(post_merged[i][j].first == merged_post[i][j].first);
      if (post_merged[i][j].second < 1e-6 && merged_post[i][j].second < 1e-6)
        continue;
      KALDI_ASSERT(kaldi::ApproxEqual(post_merged[i][j].second, merged_post[i][j].second));
    }
  }
  
  KALDI_ASSERT(ApproxEqual(merged_stats.TotalObjf(criterion), objf));
  computer->Reset();
}

}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train nnet3 neural network parameters with discriminative sequence objective \n"
        "gradient descent.  Minibatches are to be created by nnet3-discriminative-merge-egs in\n"
        "the input pipeline.  This training program is single-threaded (best to\n"
        "use it with a GPU).\n"
        "\n"
        "Usage:  nnet3-discriminative-train [options] <nnet-in> <discriminative-training-examples-in> \n"
        "\n"
        "nnet3-discriminative-train 1.mdl 'ark:nnet3-merge-egs 1.degs ark:-|' 2.raw\n";

    std::string use_gpu = "yes";
    bool compress = false;
    int32 minibatch_size = 64;

    NnetComputeProbOptions nnet_opts;
    discriminative::DiscriminativeTrainingOptions discriminative_training_opts;

    ParseOptions po(usage);
    po.Register("minibatch-size", &minibatch_size, "Target size of minibatches "
                "when merging (see also --measure-output-frames)");
    po.Register("compress", &compress, "If true, compress the output examples "
                "(not recommended unless you are writing to disk");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    nnet_opts.Register(&po);
    discriminative_training_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string model_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2);

    TransitionModel tmodel;
    AmNnetSimple am_nnet;

    bool binary;
    Input ki(model_rxfilename, &binary);
    
    tmodel.Read(ki.Stream(), binary);
    am_nnet.Read(ki.Stream(), binary);
    
    NnetDiscriminativeComputeObjf discriminative_objf_computer(nnet_opts, 
                                              discriminative_training_opts, 
                                              tmodel, am_nnet.Priors(), am_nnet.GetNnet());

    Nnet nnet = am_nnet.GetNnet();

    SequentialNnetDiscriminativeExampleReader example_reader(examples_rspecifier);

    int64 num_read = 0;

    std::vector<int32> silence_phones;
    SplitStringToIntegers(discriminative_training_opts.silence_phones_str, ":", false, &silence_phones);

    std::vector<NnetDiscriminativeExample> examples;
    while (!example_reader.Done()) {
      const NnetDiscriminativeExample &cur_eg = example_reader.Value();
      examples.resize(examples.size() + 1);
      examples.back() = cur_eg;

      bool minibatch_ready =
          static_cast<int32>(examples.size()) >= minibatch_size;

      // Do Next() now, so we can test example_reader.Done() below .
      example_reader.Next();
      num_read++;

      if (minibatch_ready || (example_reader.Done() && !examples.empty())) {
        UnitTestMergeExamples(tmodel, silence_phones, 
            discriminative_training_opts.criterion, 
            discriminative_training_opts.one_silence_class, 
            examples, &discriminative_objf_computer);
        examples.clear();
      }
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

