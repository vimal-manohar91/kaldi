// segmenterbin/agglomerative-cluster-plda.cc

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
#include "tree/clusterable-classes.h"
#include "ivector/transform.h"
#include "segmenter/iterative-bottom-up-cluster.h"
#include "segmenter/plda-clusterable.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Cluster ivectors using PLDA log-likelihood scores.\n"
      "Usage: agglomerative-cluster-plda [options] <plda> "
      "<reco2utt-rspecifier> <ivector-rspecifier> "
      "<labels-wspecifier>\n"
      "e.g.: \n"
      " agglomerative-cluster-plda plda ark:reco2utt scp:ivectors.scp \n"
      "   ark,t:labels.txt ark,t:out_utt2spk.txt\n";

    ParseOptions po(usage);
    BaseFloat target_energy = 0.5;
    std::string reco2num_spk_rspecifier, utt2num_frames_rspecifier;
    bool ivector_matrix_input = false;
    std::string thresholds_rspecifier;
    BaseFloat threshold = 0;

    po.Register("target-energy", &target_energy,
                "Reduce dimensionality of i-vectors using PCA such "
                "that this fraction of the total energy remains.");
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
    
    PldaConfig plda_config;
    plda_config.Register(&po);

    PldaClusterableOptions plda_clusterable_opts;
    plda_clusterable_opts.Register(&po);
    
    IterativeBottomUpClusteringOptions opts;
    opts.Register(&po);

    po.Read(argc, argv);
    KALDI_ASSERT(target_energy <= 1.0);

    if (po.NumArgs() != 4 && po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda_rxfilename = po.GetArg(1),
      reco2utt_rspecifier = po.GetArg(2),
      ivector_rspecifier = po.GetArg(3),
      label_wspecifier = po.GetArg(4),
      utt2spk_wspecifier = po.GetOptArg(5);

    Plda plda;
    ReadKaldiObject(plda_rxfilename, &plda);

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
    RandomAccessBaseFloatReader thresholds_reader(thresholds_rspecifier);
    Int32Writer label_writer(label_wspecifier);
    TokenWriter utt2spk_writer(utt2spk_wspecifier);

    RandomAccessInt32Reader reco2num_spk_reader(reco2num_spk_rspecifier);
    RandomAccessInt32Reader utt2num_frames_reader(utt2num_frames_rspecifier);

    int32 num_err = 0, num_done = 0;
    for (; !reco2utt_reader.Done(); reco2utt_reader.Next()) {
      const std::string &reco = reco2utt_reader.Key();
      Plda this_plda(plda);

      const std::vector<std::string> &uttlist = reco2utt_reader.Value();
      
      std::vector<std::string> out_uttlist;

      std::vector<Matrix<BaseFloat> > ivectors;
      std::vector<BaseFloat> row_weights;
      
      int32 num_ivectors = 0;
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
        
        if (!ivector_matrix_input) {
          if (!ivector_reader->HasKey(utt)) {
            KALDI_ERR << "No iVector for utterance " << utt
                      << " in archive " << ivector_rspecifier;
            num_err++;
            continue;
          }
          
          const Vector<BaseFloat> &ivector = ivector_reader->Value(utt);
          Matrix<BaseFloat> ivector_mat(1, ivector.Dim());
          ivector_mat.CopyRowFromVec(ivector, 0);
          ivectors.push_back(ivector_mat);
          num_ivectors++;
        } else {
          if (!ivector_mat_reader->HasKey(utt)) {
            KALDI_ERR << "No iVector for utterance " << utt
                      << " in archive " << ivector_rspecifier;
          }
          const Matrix<BaseFloat> &ivector_mat = ivector_mat_reader->Value(utt);
          ivectors.push_back(ivector_mat);
          num_ivectors += ivector_mat.NumRows();
        }

        out_uttlist.push_back(utt);

        row_weights.push_back(weight);
      }

      Matrix<BaseFloat> ivector_mat(num_ivectors, ivectors[0].NumCols()),
                        ivector_mat_pca, ivector_mat_plda, pca_transform;
      Vector<BaseFloat> weights(num_ivectors);
      int32 r = 0;
      for (int32 i = 0; i < ivectors.size(); i++) {
        ivector_mat.Range(r, ivectors[i].NumRows(), 
                          0, ivectors[i].NumCols()).CopyFromMat(ivectors[i]);
        weights.Range(r, ivectors[i].NumRows()).Set(row_weights[i]);
        r += ivectors[i].NumRows();
      }
      KALDI_ASSERT(r == num_ivectors);
      
      if (EstPca(ivector_mat, weights, target_energy, &pca_transform)) {
        // Apply PCA transform to the raw i-vectors.
        ApplyPca(ivector_mat, pca_transform, &ivector_mat_pca);

        // Apply PCA transform to the parameters of the PLDA model.
        this_plda.ApplyTransform(Matrix<double>(pca_transform));

        // Now transform the i-vectors using the reduced PLDA model.
        TransformIvectors(ivector_mat_pca, plda_config, this_plda,
                          &ivector_mat_plda);

        KALDI_LOG << "For recording, " << reco << " retained " 
                  << this_plda.Dim() << " dimensions.";
      } else {
        KALDI_WARN << "Unable to compute conversation dependent PCA for "
                   << "recording " << reco << ".";
        TransformIvectors(ivector_mat, plda_config, this_plda,
                          &ivector_mat_plda);
      }
      
      KALDI_ASSERT(ivectors.size() == out_uttlist.size());

      std::vector<Clusterable*> clusterables;
      r = 0;
      for (int32 i = 0; i < ivectors.size(); i++) {
        std::set<int32> points;
        points.insert(i);
        PldaClusterable *pc = new PldaClusterable(
            plda_clusterable_opts,
            &this_plda, points, 
            Vector<BaseFloat>(ivector_mat_plda.Row(r++)), row_weights[i]);

        for (int32 j = 1; j < ivectors[i].NumRows(); j++) {
          KALDI_ASSERT(r < ivector_mat_plda.NumRows());
          KALDI_ASSERT(i < row_weights.size());
          KALDI_ASSERT(pc);
          PldaClusterable other(
            plda_clusterable_opts, &this_plda, points, 
            Vector<BaseFloat>(ivector_mat_plda.Row(r++)), 
            row_weights[i]);
          pc->Add(other);


          //pc->AddStats(Vector<BaseFloat>(ivector_mat_plda.Row(r)), 
           //            row_weights[i]);
        }

        clusterables.push_back(pc);
      }
      KALDI_ASSERT(r == num_ivectors);
      
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
      
      CompartmentalizeAndClusterBottomUpPlda(
          opts, 
          (!reco2num_spk_rspecifier.empty()) ?
          std::numeric_limits<BaseFloat>::max() : this_threshold, 
          this_num_speakers,
          clusterables, NULL, &utt2cluster);

      DeletePointers(&clusterables);

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
