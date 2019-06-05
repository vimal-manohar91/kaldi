#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# This script demonstrates semi-supervised training using 50 hours of 
# supervised data and 250 hours of unsupervised data.
# This script is expected to be run after stage 8 of run_50k.sh.

# We assume the supervised data is in data/train_sup and unsupervised data
# is in data/train_unsup100k_250k. 
# For LM training, we assume there is data/train/text, from which
# we will exclude the utterances contained in the unsupervised set.
# We use all 300 hours of semi-supervised data for i-vector extractor training.

# This differs from run_lstm_100k.sh, which uses only 100 hours supervised data for 
# both i-vector extractor training and LM training.

. ./cmd.sh
. ./path.sh 

set -o pipefail
exp_root=exp/semisup_50k

stage=0

. utils/parse_options.sh

for f in data/train_sup/utt2spk data/train_unsup100k_250k/utt2spk \
  data/train/text \
  data/lang_test_poco_ex250k_big/G.carpa \
  data/lang_test_poco_ex250k/G.fst \
  data/lang_test_poco_ex250k_unk_big/G.carpa \
  data/lang_test_poco_ex250k_unk/G.fst; do
  if [ ! -f $f ]; then
    echo "$0: Could not find $f"
    exit 1
  fi
done

###############################################################################
# Train seed chain system using 50 hours supervised data.
# Here we train i-vector extractor on combined supervised and unsupervised data
###############################################################################

if [ $stage -le 1 ]; then
  local/semisup/chain/tuning/run_tdnn_lstm_1a.sh \
    --train-set train_sup50k \
    --ivector-train-set semisup50k_100k_250k \
    --nnet3-affix _semi50k_100k_250k \
    --chain-affix _semi50k_100k_250k \
    --tdnn-affix _1a --tree-affix bi_a \
    --gmm tri4a --exp-root $exp_root || exit 1
fi

###############################################################################
# Semi-supervised training using 50 hours supervised data and 
# 250 hours unsupervised data. We use i-vector extractor, tree, lattices 
# and seed chain system from the previous stage.
###############################################################################

if [ $stage -le 2 ]; then
  local/semisup/chain/tuning/run_tdnn_lstm_50k_semisupervised_1a.sh \
    --supervised-set train_sup50k \
    --unsupervised-set train_unsup100k_250k \
    --sup-chain-dir $exp_root/chain_semi50k_100k_250k/tdnn_lstm_1a_sp \
    --sup-lat-dir $exp_root/chain_semi50k_100k_250k/tri4a_train_sup50k_sp_unk_lats \
    --sup-tree-dir $exp_root/chain_semi50k_100k_250k/tree_bi_a \
    --ivector-root-dir $exp_root/nnet3_semi50k_100k_250k \
    --chain-affix _semi50k_100k_250k \
    --tdnn-affix _semisup_1a \
    --exp-root $exp_root || exit 1
fi

###############################################################################
# Oracle system trained on combined 300 hours including both supervised and 
# unsupervised sets. We use i-vector extractor, tree, and GMM trained
# on only the supervised for fair comparison to semi-supervised experiments.
###############################################################################

if [ $stage -le 3 ]; then
  local/semisup/chain/tuning/run_tdnn_lstm_1b.sh \
    --train-set semisup50k_100k_250k \
    --nnet3-affix _semi50k_100k_250k \
    --chain-affix _semi50k_100k_250k \
    --common-treedir $exp_root/chain_semi50k_100k_250k/tree_bi_a \
    --tdnn-affix _1b_oracle --nj 100 \
    --gmm tri4a --exp-root $exp_root \
    --stage 9 || exit 1
fi
