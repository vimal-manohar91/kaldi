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

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Cluster ivectors using euclidean distance\n"
      "TODO better documentation\n"
      "Usage: agglomerative-cluster-ivectors [options] "
      "<reco2utt-rspecifier> <ivectors-rspecifier> <labels-wspecifier> [<utt2spk-wspecifier>]\n"
      "e.g.: \n"
      " agglomerative-cluster-ivectors ark:reco2utt ark:ivectors.ark \n"
      "   ark,t:labels.txt ark,t:out_utt2spk.txt\n";

    ParseOptions po(usage);
    std::string reco2num_spk_rspecifier, utt2num_frames_rspecifier;
    bool ivector_matrix_input = false;
    std::string thresholds_rspecifier;
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
    po.Register("thresholds-rspecifier", &thresholds_rspecifier,
                "If specified, applies a per-recording threshold; "
                "overrides --threshold.");

    IvectorClusterableOptions ivector_clusterable_opts;
    ivector_clusterable_opts.Register(&po);

    IterativeBottomUpClusteringOptions opts;
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3 && po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string reco2utt_rspecifier = po.GetArg(1),
      ivectors_rspecifier = po.GetArg(2),
      label_wspecifier = po.GetArg(3),
      utt2spk_wspecifier = po.GetOptArg(4);

    SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);

    RandomAccessBaseFloatVectorReader *ivector_reader = NULL;
    RandomAccessBaseFloatMatrixReader *ivector_mat_reader = NULL;
    if (ivector_matrix_input) {
      ivector_mat_reader = 
        new RandomAccessBaseFloatMatrixReader(ivectors_rspecifier);
    } else {
      ivector_reader =
        new RandomAccessBaseFloatVectorReader(ivectors_rspecifier);
    }
    RandomAccessBaseFloatReader thresholds_reader(thresholds_rspecifier);
    Int32Writer label_writer(label_wspecifier);
    TokenWriter utt2spk_writer(utt2spk_wspecifier);

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
        
        IvectorClusterable *ic;

        if (!ivector_matrix_input) {
          if (!ivector_reader->HasKey(utt)) {
            KALDI_ERR << "No iVector for utterance " << utt
                      << " in archive " << ivectors_rspecifier;
            num_err++;
            continue;
          }
          
          const Vector<BaseFloat> &ivector = ivector_reader->Value(utt);

          ic = new IvectorClusterable(ivector_clusterable_opts, points, 
                                      ivector, weight);
        } else {
          if (!ivector_mat_reader->HasKey(utt)) {
            KALDI_ERR << "No iVector for utterance " << utt
                      << " in archive " << ivectors_rspecifier;
            num_err++;
            continue;
          }
          const Matrix<BaseFloat> &ivector_mat = ivector_mat_reader->Value(utt);

          KALDI_ASSERT(ivector_mat.NumRows() > 0);
          ic = new IvectorClusterable(ivector_clusterable_opts, points, 
                                      Vector<BaseFloat>(ivector_mat.Row(0)), 
                                      weight);
          for (int32 r = 1; r < ivector_mat.NumRows(); r++) {
            IvectorClusterable this_ic(
                points, Vector<BaseFloat>(ivector_mat.Row(r)), weight);
            ic->Add(this_ic);
          }
        }

        out_uttlist.push_back(utt);
        clusterables.push_back(ic);
      }

      BaseFloat this_threshold = threshold;
      if (!thresholds_rspecifier.empty()) {
        if (!thresholds_reader.HasKey(reco)) {
          KALDI_WARN << "Could not find threshold for recording " << reco 
                     << " in " << thresholds_rspecifier << "; using "
                     << "--threshold=" << threshold;
        } else {
          this_threshold = thresholds_reader.Value(reco);
        }
      } 

      int32 this_num_speakers = 1;
      if (!reco2num_spk_rspecifier.empty()) {
        this_num_speakers = reco2num_spk_reader.Value(reco);
      } 

      std::vector<int32> utt2cluster(out_uttlist.size());

      CompartmentalizeAndClusterBottomUpIvector(
          opts, 
          (!reco2num_spk_rspecifier.empty()) ?
          std::numeric_limits<BaseFloat>::max() : this_threshold, 
          this_num_speakers,
          clusterables, NULL, &utt2cluster);

      for (size_t i = 0; i < out_uttlist.size(); i++) {
        label_writer.Write(out_uttlist[i], utt2cluster[i]);
      }
      
      if (!utt2spk_wspecifier.empty()) {
        for (size_t i = 0; i < out_uttlist.size(); i++) {
          std::ostringstream oss;
          oss << reco << "-" << utt2cluster[i];
          utt2spk_writer.Write(out_uttlist[i], oss.str());
        }
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
