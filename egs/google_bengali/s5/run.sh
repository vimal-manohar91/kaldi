#!/bin/bash
# Copyright (c) 2018, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
stage=0
corpus=./corpus
nj=16
dev_nj=6

# End configuration section
. ./utils/parse_options.sh

# initialization PATH
. ./path.sh  || die "File path.sh expected";
. ./cmd.sh  || die "File cmd.sh expected to exist"

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

if [ $stage -le 0 ]; then
  ./local/download_data.sh --datadir $corpus
fi

if [ $stage -le 1 ]; then
  echo "Preparing data and training language models"
  local/prepare_data.sh $corpus/
fi

if [ $stage -le 2 ]; then
  . /opt/anaconda3/etc/profile.d/conda.sh
  conda activate
  local/prepare_dict.sh $corpus/
  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
  conda deactivate
fi

if [ $stage -le 3 ]; then
  local/prepare_lm.sh --stage 4 $corpus/
fi

if [ $stage -le 4 ]; then
  # Feature extraction
  for x in train dev; do
      steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/$x exp/make_mfcc/$x mfcc
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc
  done
fi

if [ $stage -le 5 ]; then
  ### Monophone
  echo "Starting monophone training."
  utils/subset_data_dir.sh data/train 1000 data/train.1k
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" data/train.1k data/lang exp/mono
  echo "Mono training done."

	# this is not run parallel intentionally
  utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph
  (
  echo "Decoding the dev set using monophone models."

  steps/decode.sh --config conf/decode.config --nj $dev_nj --cmd "$decode_cmd" \
    exp/mono/graph data/dev exp/mono/decode_dev
  echo "Monophone decoding done."
  ) &
fi


if [ $stage -le 6 ]; then
  ### Triphone
  echo "Starting triphone training."
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
      data/train data/lang exp/mono exp/mono_ali
  steps/train_deltas.sh --boost-silence 1.25  --cmd "$train_cmd"  \
      3200 30000 data/train data/lang exp/mono_ali exp/tri1
  echo "Triphone training done."

  (
  echo "Decoding the dev set using triphone models."
  utils/mkgraph.sh data/lang_test  exp/tri1 exp/tri1/graph
  steps/decode.sh --nj $dev_nj --cmd "$decode_cmd"  \
      exp/tri1/graph  data/dev exp/tri1/decode_dev

  steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" \
      data/lang_test/ data/lang_test_fg/ data/dev \
      exp/tri1/decode_dev exp/tri1/decode_dev.rescored
  echo "Triphone decoding done."
  ) &
fi

if [ $stage -le 7 ]; then
  ## Triphones + delta delta
  # Training
  echo "Starting (larger) triphone training."
  steps/align_si.sh  --nj $nj --cmd "$train_cmd" --use-graphs true \
       data/train data/lang exp/tri1 exp/tri1_ali
  steps/train_deltas.sh --cmd "$train_cmd"  \
      4200 40000 data/train data/lang exp/tri1_ali exp/tri2a
  echo "Triphone (large) training done."

  (
  echo "Decoding the dev set using triphone(large) models."
  utils/mkgraph.sh data/lang_test  exp/tri2a exp/tri2a/graph
  steps/decode.sh --nj $dev_nj --cmd "$decode_cmd" \
      exp/tri2a/graph  data/dev exp/tri2a/decode_dev

  steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" \
      data/lang_test/ data/lang_test_fg/ data/dev \
      exp/tri2a/decode_dev exp/tri2a/decode_dev.rescored
  echo "Triphone(large) decoding done."
  ) &
fi

if [ $stage -le 8 ]; then
  ### Triphone + LDA and MLLT
  # Training
  echo "Starting LDA+MLLT training."
  steps/align_si.sh  --nj $nj --cmd "$train_cmd"  \
      data/train data/lang exp/tri2a exp/tri2a_ali

  steps/train_lda_mllt.sh  --cmd "$train_cmd"  \
    --splice-opts "--left-context=3 --right-context=3" \
    4200 40000 data/train data/lang  exp/tri2a_ali exp/tri2b
  echo "LDA+MLLT training done."

  (
  echo "Decoding the dev set using LDA+MLLT models."
  utils/mkgraph.sh data/lang_test  exp/tri2b exp/tri2b/graph
  steps/decode.sh --nj $dev_nj    --cmd "$decode_cmd" \
      exp/tri2b/graph  data/dev exp/tri2b/decode_dev

  steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" \
      data/lang_test/ data/lang_test_fg/ data/dev \
      exp/tri2b/decode_dev exp/tri2b/decode_dev.rescored
  echo "LDA+MLLT decoding done."
  ) &
fi


if [ $stage -le 9 ]; then
  ### Triphone + LDA and MLLT + SAT and FMLLR
  # Training
  echo "Starting SAT+FMLLR training."
  steps/align_si.sh  --nj $nj --cmd "$train_cmd" \
      --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali
  steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
      data/train data/lang exp/tri2b_ali exp/tri3b
  echo "SAT+FMLLR training done."

  (
  echo "Decoding the dev set using SAT+FMLLR models."
  utils/mkgraph.sh data/lang_test  exp/tri3b exp/tri3b/graph
  steps/decode_fmllr.sh --nj $dev_nj --cmd "$decode_cmd" \
      exp/tri3b/graph  data/dev exp/tri3b/decode_dev

  steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" \
      data/lang_test/ data/lang_test_fg/ data/dev \
      exp/tri3b/decode_dev exp/tri3b/decode_dev.rescored
  echo "SAT+FMLLR decoding done."
  ) &
fi


if [ $stage -le 10 ]; then
  echo "Starting SGMM training."
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/train data/lang exp/tri3b exp/tri3b_ali

  steps/train_ubm.sh  --cmd "$train_cmd"  \
      600 data/train data/lang exp/tri3b_ali exp/ubm5b2

  steps/train_sgmm2.sh  --cmd "$train_cmd"  \
       5200 12000 data/train data/lang exp/tri3b_ali exp/ubm5b2/final.ubm exp/sgmm2_5b2
  echo "SGMM training done."

  (
  echo "Decoding the dev set using SGMM models"
  # Graph compilation
  utils/mkgraph.sh data/lang_test exp/sgmm2_5b2 exp/sgmm2_5b2/graph

  steps/decode_sgmm2.sh --nj $dev_nj --cmd "$decode_cmd" \
      --transform-dir exp/tri3b/decode_dev \
      exp/sgmm2_5b2/graph data/dev exp/sgmm2_5b2/decode_dev

  steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" \
      data/lang_test/ data/lang_test_fg/ data/dev \
      exp/sgmm2_5b2/decode_dev exp/sgmm2_5b2/decode_dev.rescored

  echo "SGMM decoding done."
  ) &
  # this is extremely computationally and memory-wise expensive, run with caution
  # or just don't run at all, there is no practical benefit
  #-(
  #-echo "Decoding the dev set using SGMM models and LargeLM"
  #-# Graph compilation
  #-utils/mkgraph.sh data/lang_test_fg/ exp/sgmm2_5b2 exp/sgmm2_5b2/graph_big

  #-steps/decode_sgmm2.sh --nj $dev_nj --cmd "$decode_cmd" \
  #-    --transform-dir exp/tri3b/decode_dev \
  #-    exp/sgmm2_5b2/graph_big data/dev exp/sgmm2_5b2/decode_dev.big
  #-echo "SGMM decoding done."
  #-) &
fi

  (
  echo "Decoding the dev set using SGMM models"
  # Graph compilation
  #utils/mkgraph.sh data/lang_test exp/sgmm2_5b2 exp/sgmm2_5b2/graph

  steps/decode_sgmm2.sh --nj $dev_nj --cmd "$decode_cmd" \
      --transform-dir exp/tri3b/decode_dev \
      exp/sgmm2_5b2/graph data/dev exp/sgmm2_5b2/decode_dev

  steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" \
      data/lang_test/ data/lang_test_fg/ data/dev \
      exp/sgmm2_5b2/decode_dev exp/sgmm2_5b2/decode_dev.rescored

  echo "SGMM decoding done."
  ) &
wait;
#score
find exp -name "best_wer" | xargs cat  | sort -k2,2g

# to run nnet3 model and chain model (notice the parameter --stage 9, that ensures the ivector wont get
# overwritten and that certain portions of the training will be shared accross both models
# ./local/nnet3/run_tdnn.sh
# ./local/chain/run_tdnn.sh --stage 9

# if you want to run chain system only or "normal" tdnn system only,
# run either
# ./local/nnet3/run_tdnn.sh
# or
# ./local/chain/run_tdnn.sh
# (without the stage parameter)

# you can check the scores again using
# find exp -name "best_wer" | xargs cat  | sort -k2,2g
