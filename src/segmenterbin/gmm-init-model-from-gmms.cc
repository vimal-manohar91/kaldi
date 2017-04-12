// gmmbin/gmm-init-model-from-gmms.cc

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
#include "hmm/hmm-topology.h"
#include "hmm/transition-model.h"
#include "gmm/am-diag-gmm.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Initialize simple HMM-GMM acoustic model from GMMs\n"
        "Usage:  gmm-init-from-gmms [options] <topology-in> "
        "<gmms-rspecifier> <model-out>\n"
        "e.g.: \n"
        "  gmm-init-from-gmms trans.mdl ark:gmms.ark 1.mdl\n";

    bool binary = true;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string topo_filename = po.GetArg(1),
      gmms_rspecifier = po.GetArg(2),
      model_out_filename = po.GetArg(3),
      tree_filename = po.GetArg(4);

    HmmTopology topo;
    ReadKaldiObject(topo_filename, &topo);

    SequentialDiagGmmReader gmms_reader(gmms_rspecifier);

    AmDiagGmm am_gmm;

    std::vector<DiagGmm> gmms;

    int32 num_pdfs = 0;

    KALDI_ASSERT(topo.GetPhones().size() == 1);
    int32 base_phone = topo.GetPhones().back();
    KALDI_ASSERT(topo.NumPdfClasses(base_phone) == 1);

    for (; !gmms_reader.Done(); gmms_reader.Next(), num_pdfs++) {
      const std::string &key = gmms_reader.Key();

      if (num_pdfs > 0) {
        topo.CloneTopology(base_phone);
      }

      am_gmm.AddPdf(gmms_reader.Value());
    }

    const std::vector<int32> &phones = topo.GetPhones();
    KALDI_ASSERT(phones.size() == num_pdfs);
    std::vector<int32> phone2num_pdf_classes (1+phones.back());
    for (size_t i = 0; i < phones.size(); i++)
      phone2num_pdf_classes[phones[i]] = topo.NumPdfClasses(phones[i]);

    ContextDependency *ctx_dep = MonophoneContextDependency(
        phones, phone2num_pdf_classes);

    TransitionModel trans_model(*ctx_dep, topo);

    {
      Output ko(model_out_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_gmm.Write(ko.Stream(), binary);
    }

    ctx_dep->Write(Output(tree_filename, binary).Stream(), binary);

    delete ctx_dep;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
