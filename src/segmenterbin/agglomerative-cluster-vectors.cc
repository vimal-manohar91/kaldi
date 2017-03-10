// ivectorbin/agglomerative-cluster-vectors.cc

// Copyright 2017  Vimal Manohar

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
#include "util/stl-utils.h"
#include "tree/cluster-utils.h"
#include "tree/clusterable-classes.h"
#include "segmenter/iterative-bottom-up-cluster.h"
#include "segmenter/ivector-clusterable.h"
namespace kaldi {

void CompartmentalizeAndClusterBottomUp(
    const IterativeBottomUpClusteringOptions &opts,
    BaseFloat max_merge_thresh, int32 min_clust,
    const std::vector<Clusterable*> &points,
    std::vector<Clusterable*> *clusters_out,
    std::vector<int32> *assignments_out) {
  IterativeBottomUpClusterer<IvectorClusterable> clusterer(
      opts, max_merge_thresh, min_clust, points, 
      clusters_out, assignments_out);
  clusterer.Cluster();
}

}  // end namespace kaldi

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Cluster ivectors using euclidean distance\n"
      "TODO better documentation\n"
      "Usage: agglomerative-cluster-ivectors [options] <ivectors-rspecifier> "
      "<reco2utt-rspecifier> <labels-wspecifier>\n"
      "e.g.: \n"
      " agglomerative-cluster-ivectors ark:ivectors.ark ark:reco2utt \n"
      "   ark,t:labels.txt\n";

    ParseOptions po(usage);
    std::string reco2num_spk_rspecifier, utt2num_frames_rspecifier;
    BaseFloat threshold = 0.5;

    po.Register("reco2num-spk-rspecifier", &reco2num_spk_rspecifier,
                "If supplied, clustering creates exactly this many clusters "
                "for each recording and the option --threshold is ignored.");
    po.Register("utt2num-frames-rspecifier", &utt2num_frames_rspecifier,
                "The number of frames in each utterance.");
    po.Register("threshold", &threshold, 
                "Merging clusters if their distance"
                "is less than this threshold.");
    po.Register("ivector-matrix-input", &ivector_matrix_input,
                "If true, expects i-vector input as a matrix with "
                "possibly multiple i-vectors per utterance.");
    IterativeBottomUpClusteringOptions opts;
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string ivectors_rspecifier = po.GetArg(1),
      reco2utt_rspecifier = po.GetArg(2),
      label_wspecifier = po.GetArg(3);

    SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);

    RandomAccessBaseFloatVectorReader *ivector_reader = NULL;
    RandomAccessBaseFloatMatrixReader *ivector_mat_reader = NULL;
    if (ivector_matrix_input) {
      ivector_mat_reader = 
        new RandomAccessBaseFloatMatrixReader(ivector_rspecifier);
    } else {
      ivector_reader =
        new RandomAccessBaseFloatVectorReader(ivector_rspecifier);
    }
    Int32Writer label_writer(label_wspecifier);

    RandomAccessInt32Reader reco2num_spk_reader(reco2num_spk_rspecifier);
    RandomAccessInt32Reader utt2num_frames_reader(utt2num_frames_rspecifier);

    int32 num_err = 0, num_done = 0;
    for (; !reco2utt_reader.Done(); reco2utt_reader.Next()) {
      const std::string &reco = reco2utt_reader.Key();
      const std::vector<std::string> &uttlist = reco2utt_reader.Value();
      
      std::vector<std::string> out_uttlist;

      std::vector<Clusterable*> clusterables;
      for (size_t i = 0; i < uttlist.size(); i++) {
        const std::string &utt = uttlist[i];
        
        int32 weight = 1;
        if (!utt2num_frames_rspecifier.empty()) {
          if (!utt2num_frames_reader.HasKey(utt)) {
            KALDI_ERR << "No weights for utterance " << utt << " in archive " 
                      << utt2num_frames_rspecifier;
            num_err++;
            continue;
          }
          weight = utt2num_frames_reader.Value(utt);
        }
        
        std::set<int32> points;
        points.insert(i);
        
        IvectorClusterable *ic = new IvectorClusterable(points);

        if (!ivector_matrix_input) {
          if (!ivector_reader->HasKey(utt)) {
            KALDI_ERR << "No iVector for utterance " << utt
                      << " in archive " << ivector_rspecifier;
            num_err++;
            continue;
          }
          
          const Vector<BaseFloat> &ivector = ivector_reader->Value(utt);
          ic->AddStats(ivector, weight);

        } else {
          if (!ivector_mat_reader->HasKey(utt)) {
            KALDI_ERR << "No iVector for utterance " << utt
                      << " in archive " << ivector_rspecifier;
          }
          const Matrix<BaseFloat> &ivector_mat = ivector_mat_reader->Value(utt);

          KALDI_ASSERT(ivector_mat.NumRows() > 0);
          for (int32 r = 0; r < ivector_mat.NumRows(); r++) {
            ic->AddStats(ivector_mat.Row(i), weight);
          }
        }

        clusterables.push_back(ic);
      }

      int32 this_num_speakers = 1;
      if (!reco2num_spk_rspecifier.empty()) {
        this_num_speakers = reco2num_spk_reader.Value(reco);
      } 

      std::vector<int32> utt2cluster(out_uttlist.size());

      CompartmentalizeAndClusterBottomUp(
          opts, 
          (!reco2num_spk_rspecifier.empty()) ?
          std::numeric_limits<BaseFloat>::max() : threshold, 
          this_num_speakers,
          clusterables, NULL, &utt2cluster);

      for (size_t i = 0; i < out_uttlist.size(); i++) {
        label_writer.Write(out_uttlist[i], utt2cluster[i]);
      }

      num_done++;
    }
    
    delete ivector_reader;
    delete ivector_mat_reader;

    return (num_done > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
