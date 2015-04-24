// nnet2/nnet-compute-discriminative-unsupervised-parallel.cc

// Copyright 2015   Vimal Manohar (Johns Hopkins University)

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

#include <deque>
#include "nnet2/nnet-compute-discriminative-unsupervised-parallel.h"
#include "hmm/posterior.h"
#include "lat/lattice-functions.h"
#include "thread/kaldi-semaphore.h"
#include "thread/kaldi-mutex.h"
#include "thread/kaldi-thread.h"

namespace kaldi {
namespace nnet2 {
  
/** This struct stores neural net training examples to be used in
    multi-threaded training.  */
class DiscriminativeUnsupervisedExamplesRepository {
  public:
    /// The following function is called by the code that reads in the examples.
    void AcceptExample(const DiscriminativeUnsupervisedNnetExample &example);

    /// The following function is called by the code that reads in the examples,
    /// when we're done reading examples; it signals this way to this class
    /// that the stream is now empty
    void ExamplesDone();

    /// This function is called by the code that does the training. If there is
    /// an example available it will provide it, or it will sleep till one is 
    /// availabe. It returns NULL when there are no examples left and
    /// ExamplesDone() has been called.
    DiscriminativeUnsupervisedNnetExample *ProvideExample();

    DiscriminativeUnsupervisedExamplesRepository(): buffer_size_(4), 
    empty_semaphore_(buffer_size_),
    done_(false) { }

  private:
    int32 buffer_size_;
    Semaphore full_semaphore_;
    Semaphore empty_semaphore_;
    Mutex examples_mutex_;  // mutex we lock to modify examples_.

    std::deque<DiscriminativeUnsupervisedNnetExample*> examples_;
    bool done_;

    KALDI_DISALLOW_COPY_AND_ASSIGN(DiscriminativeUnsupervisedExamplesRepository);
};

void DiscriminativeUnsupervisedExamplesRepository::AcceptExample(
    const DiscriminativeUnsupervisedNnetExample &example) {
  empty_semaphore_.Wait();
  examples_mutex_.Lock();
  examples_.push_back(new DiscriminativeUnsupervisedNnetExample(example));
  examples_mutex_.Unlock();
  full_semaphore_.Signal();
}

void DiscriminativeUnsupervisedExamplesRepository::ExamplesDone() {
  for (int32 i = 0; i < buffer_size_; i++)
    empty_semaphore_.Wait();
  examples_mutex_.Lock();
  KALDI_ASSERT(examples_.empty());
  examples_mutex_.Unlock();
  done_ = true;
  full_semaphore_.Signal();
}

DiscriminativeUnsupervisedNnetExample* DiscriminativeUnsupervisedExamplesRepository::ProvideExample() {
  full_semaphore_.Wait();
  if (done_) {
    KALDI_ASSERT(examples_.empty());
    full_semaphore_.Signal(); // Increment the semaphore so
    // the call by the next thread will not block.
    return NULL; // no examples to return-- all finished.
  } else {
    examples_mutex_.Lock();
    KALDI_ASSERT(!examples_.empty());
    DiscriminativeUnsupervisedNnetExample *ans = examples_.front();
    examples_.pop_front();
    examples_mutex_.Unlock();
    empty_semaphore_.Signal();
    return ans;
  }
}


class DiscriminativeUnsupervisedTrainParallelClass: public MultiThreadable {
  public:
    // This constructor is only called for a temporary object
    // that we pass to the RunMultiThreaded function.
    DiscriminativeUnsupervisedTrainParallelClass(const AmNnet &am_nnet,
        const TransitionModel &tmodel,
        const NnetDiscriminativeUnsupervisedUpdateOptions &opts,
        bool store_separate_gradients,
        DiscriminativeUnsupervisedExamplesRepository *repository,
        Nnet *nnet_to_update,
        NnetDiscriminativeUnsupervisedStats *stats):
      am_nnet_(am_nnet), tmodel_(tmodel), opts_(opts),
      store_separate_gradients_(store_separate_gradients),
      repository_(repository),
      nnet_to_update_(nnet_to_update),
      nnet_to_update_orig_(nnet_to_update),
      stats_ptr_(stats) {
        if(stats->store_gradients) {
          KALDI_ASSERT(stats->gradients.Dim() > 0);
          stats_.gradients.Resize((stats->gradients).Dim());
          stats_.output.Resize((stats->output).Dim());
          stats_.store_gradients = true;
        }
      }

    // The following constructor is called multiple times within
    // the RunMultiThreaded template function.
    DiscriminativeUnsupervisedTrainParallelClass(
        const DiscriminativeUnsupervisedTrainParallelClass &other):
      am_nnet_(other.am_nnet_), tmodel_(other.tmodel_), opts_(other.opts_),
      store_separate_gradients_(other.store_separate_gradients_),
      repository_(other.repository_), nnet_to_update_(other.nnet_to_update_),
      nnet_to_update_orig_(other.nnet_to_update_orig_),
      stats_ptr_(other.stats_ptr_) {
        if (store_separate_gradients_) {
          // To ensure correctness, we work on separate copies of the gradient
          // object, which we'll sum at the end. This is used for exact gradient
          // computation.
          if (other.nnet_to_update_ != NULL) {
            nnet_to_update_ = new Nnet(*(other.nnet_to_update_));
            // our "nnet_to_update_" variable is a copy of the neural network
            // we are to update (presumably a gradient). If we don't set these
            // to zero we would end up adding multiple copies of any initial
            // gradient that "nnet_to_update_" contained when we initialize
            // the first instance of the class.
            nnet_to_update_->SetZero(true);
          } else { // support case where we don't really need a gradient.
            nnet_to_update_ = NULL;
          }
        }
        
        if(other.stats_.store_gradients) {
          KALDI_ASSERT(other.stats_.gradients.Dim() > 0);
          stats_.gradients.Resize((other.stats_.gradients).Dim());
          stats_.output.Resize((other.stats_.output).Dim());
          stats_.store_gradients = true;
        }
      }
    // This doesthe main function of the class.
    void operator () () {
      DiscriminativeUnsupervisedNnetExample *example;
      while ((example = repository_->ProvideExample()) != NULL) {
        // This is a function call to a function defined in 
        // nnet-compute-discriminative-unsupervised.h
        NnetDiscriminativeUnsupervisedUpdate(am_nnet_, tmodel_, opts_,
            *example, nnet_to_update_,
            &stats_);
        delete example;

        if (GetVerboseLevel() > 3) {
          KALDI_VLOG(3) << "Printing local stats for thread " << thread_id_;
          stats_.Print(opts_.criterion);
        }
      }
    }

    ~DiscriminativeUnsupervisedTrainParallelClass() {
      if (nnet_to_update_orig_ != nnet_to_update_) {
        // This branch is only taken if this instance of the class is
        // one of the multiple instance allocated inside the RunMultiThreaded
        // template function, *and* store_separate_gradients_ has been set to true.
        // In this typical hogwild case, we don't do this.
        nnet_to_update_orig_->AddNnet(1.0, *nnet_to_update_);
        delete nnet_to_update_;
      }
      stats_ptr_->Add(stats_);
    }
  private:
    const AmNnet &am_nnet_;
    const TransitionModel &tmodel_;
    const NnetDiscriminativeUnsupervisedUpdateOptions &opts_;
    bool store_separate_gradients_;
    DiscriminativeUnsupervisedExamplesRepository *repository_;
    Nnet *nnet_to_update_;
    Nnet *nnet_to_update_orig_;
    NnetDiscriminativeUnsupervisedStats *stats_ptr_;
    NnetDiscriminativeUnsupervisedStats stats_;
};


void NnetDiscriminativeUnsupervisedUpdateParallel(
    const AmNnet &am_nnet,
    const TransitionModel &tmodel,
    const NnetDiscriminativeUnsupervisedUpdateOptions &opts,
    int32 num_threads,
    SequentialDiscriminativeUnsupervisedNnetExampleReader *example_reader,
    Nnet *nnet_to_update,
    NnetDiscriminativeUnsupervisedStats *stats) {

  DiscriminativeUnsupervisedExamplesRepository repository;

  const bool store_separate_gradients = (nnet_to_update != &(am_nnet.GetNnet()));

  DiscriminativeUnsupervisedTrainParallelClass c(am_nnet, tmodel, opts,
                                        store_separate_gradients,
                                        &repository, nnet_to_update,
                                        stats);

  {
    // The initialization of the following class spawns the threads that 
    // process the examples. They get re-joined in its destructor.
    MultiThreader<DiscriminativeUnsupervisedTrainParallelClass> m(
        num_threads, c);

    for (; !example_reader->Done(); example_reader->Next()) {
      repository.AcceptExample(example_reader->Value());
    }
    repository.ExamplesDone();
  }
  stats->Print(opts.criterion);
}



} // namespace nnet2
} // namespace kaldi
