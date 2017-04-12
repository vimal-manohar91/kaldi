// ivectorbin/ivector-transform-plda.cc

// Copyright 2013  Daniel Povey
//           2017  Vimal Manohar

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
#include "ivector/plda.h"
#include "ivector/transform.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Transforms iVectors by the PLDA transformaion\n"
        "\n"
        "Usage:  ivector-transform-plda [options] <plda> [<reco2utt>] <ivector-rspecifier>"
        "<ivector-wspecifier>\n"
        "e.g.: \n"
        " ivector-transform-plda plda reco2utt ark:ivectors.ark ark:transformed_ivectors.ark\n";
    
    ParseOptions po(usage);
    BaseFloat target_energy = 0.5;
    std::string utt2num_examples_rspecifier;
    bool ivector_matrix_input = false;

    PldaConfig plda_config;
    plda_config.Register(&po);
    po.Register("target-energy", &target_energy,
                "Reduce dimensionality of i-vectors using PCA such "
                "that this fraction of the total energy remains.");
    po.Register("utt2num-examples", &utt2num_examples_rspecifier, 
                "Table to read the number of examples (segments) used to "
                "estimate iVector.");
    po.Register("ivector-matrix-input", &ivector_matrix_input,
                "If true, expects i-vector input as a matrix with "
                "possibly multiple i-vectors per utterance.");

    po.Read(argc, argv);
    KALDI_ASSERT(target_energy <= 1.0);
    
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda_rxfilename = po.GetArg(1),
        reco2utt_rspecifier = po.GetArg(2),
        ivector_rspecifier = po.GetArg(3),
        ivector_wspecifier = po.GetArg(4);

    Plda plda;
    ReadKaldiObject(plda_rxfilename, &plda);
    
    SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
    RandomAccessBaseFloatVectorReader *ivector_reader = NULL;
    BaseFloatVectorWriter *ivector_writer = NULL;
    RandomAccessBaseFloatMatrixReader *ivector_mat_reader = NULL;
    BaseFloatMatrixWriter *ivector_mat_writer = NULL;

    if (!ivector_matrix_input) {
      ivector_reader = new RandomAccessBaseFloatVectorReader(ivector_rspecifier);
      ivector_writer = new BaseFloatVectorWriter(ivector_wspecifier);
    } else {
      ivector_mat_reader = new RandomAccessBaseFloatMatrixReader(ivector_rspecifier);
      ivector_mat_writer = new BaseFloatMatrixWriter(ivector_wspecifier);
    }

    RandomAccessInt32Reader utt2num_examples_reader(
        utt2num_examples_rspecifier);
    
    int32 num_reco_err = 0, num_reco_done = 0;

    for (; !reco2utt_reader.Done(); reco2utt_reader.Next()) {
      const std::string &reco = reco2utt_reader.Key();
      Plda this_plda(plda);

      const std::vector<std::string> &uttlist = reco2utt_reader.Value();

      std::vector<Matrix<BaseFloat> > ivectors;
      std::vector<BaseFloat> row_weights;

      int32 num_ivectors = 0;
      for (size_t i = 0; i < uttlist.size(); i++) {
        const std::string &utt = uttlist[i];

        int32 weight = 1;
        if (!utt2num_examples_rspecifier.empty()) {
          if (!utt2num_examples_reader.HasKey(utt)) {
            KALDI_ERR << "No weights for utterance " << utt << " in archive " 
                      << utt2num_examples_rspecifier;
          }
          weight = utt2num_examples_reader.Value(utt);
        }
        
        if (!ivector_matrix_input) {
          if (!ivector_reader->HasKey(utt)) {
            KALDI_ERR << "No iVector for utterance " << utt
                      << " in archive " << ivector_rspecifier;
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
        row_weights.push_back(weight);
      }

      Matrix<BaseFloat> ivector_mat(num_ivectors, ivectors[0].NumCols()),
                        ivector_mat_pca, ivector_mat_plda, pca_transform;
      Vector<BaseFloat> weights(num_ivectors);
      for (int32 i = 0, r = 0; i < ivectors.size(); 
           i++, r += ivectors[i].NumRows()) {
        ivector_mat.Range(r, ivectors[i].NumRows(), 
                          0, ivectors[i].NumCols()).CopyFromMat(ivectors[i]);
        weights.Range(r, ivectors[i].NumRows()).Set(row_weights[i]);
      }
      
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
      
      for (int32 i = 0, r = 0; i < ivectors.size(); 
           i++, r += ivectors[i].NumRows()) {
        if (!ivector_matrix_input) {
          ivector_writer->Write(uttlist[i], 
                                Vector<BaseFloat>(ivector_mat_plda.Row(r)));
        } else {
          SubMatrix<BaseFloat> this_mat(
              ivector_mat_plda, r, ivectors[i].NumRows(),
              0, ivectors[i].NumCols());
          ivector_mat_writer->Write(uttlist[i], Matrix<BaseFloat>(this_mat));
        }
      }

      num_reco_done++;
    }
    
    delete ivector_reader;
    delete ivector_writer;
    delete ivector_mat_reader;
    delete ivector_mat_writer;

    KALDI_LOG << "Processed " << num_reco_done << " recordings, "
              << num_reco_err << " had errors.";
    return (num_reco_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

