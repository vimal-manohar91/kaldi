#!/bin/bash

. ./cmd.sh
. ./path.sh

false && {
utils/subset_data_dir.sh --speakers data/train 300000 data/train_300k

# First 10000 sentences corresponding to dev_and_test will be used for 
# reporting perplexities
utils/combine_data.sh data/train_300k_dev data/dev data/test data/train_300k

local/fisher_train_lms_pocolm.sh --text data/train_300k_dev/text \
  --lexicon data/local/dict/lexicon.txt --dir data/local/pocolm_300k --num-ngrams-big 250000

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" --stage 2 \
  data/train_300k data/lang exp/tri4a exp/semisup300k/tri4a_ali

steps/train_sat.sh --stage 18 --cmd "$train_cmd" 7000 200000 \
  data/train_300k data/lang exp/semisup300k/tri4a_ali exp/semisup300k/tri5b

local/semisup/build_silprob.sh

local/fisher_train_lms_pocolm.sh --text data/train_300k/text \
  --lexicon data/local/dict/lexicon.txt \
  --dir data/local/pocolm_300k \
  --num-ngrams-large 250000

local/fisher_create_test_lang.sh \
  --arpa-lm data/local/pocolm_300k/data/arpa/4gram_big.arpa.gz \
  --lang data/lang_300k_pp \
  --dir data/lang_300k_pp_test

local/semisup/chain/tuning/run_tdnn_lstm_300k_1a.sh \
  --train-set train_300k --exp exp/semisup300k \
  --gmm tri5b --stage 15

local/semisup/chain/tuning/run_tdnn_lstm_300k_norvb_1a.sh \
  --train-set train_300k --exp exp/semisup300k \
  --gmm tri5b --stage 15
}

local/semisup/chain/tuning/run_tdnn_lstm_300k_1b.sh \
  --train-set train_300k --exp exp/semisup300k \
  --gmm tri5b --stage 17 --train-stage 244
