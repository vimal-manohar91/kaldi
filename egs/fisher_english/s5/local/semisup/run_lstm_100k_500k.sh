#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# This script demonstrates semi-supervised training using 100 hours of 
# supervised data and 250 hours of unsupervised data.
# We assume the supervised data is in data/train_sup and unsupervised data
# is in data/train_unsup100k_250k. 
# For LM training, we only use the supervised set corresponding to 100 hours as 
# opposed to the case in run_50k.sh, where we included part of the 
# transcripts in data/train/text.
# This uses only 100 hours supervised set for i-vector extractor training, 
# which is different from run_50k.sh, which uses combined supervised + 
# unsupervised set.
# This script is expected to be run after stage 8 of run_100k.sh.

. ./cmd.sh
. ./path.sh 

set -o pipefail
exp_root=exp/semisup_100k

stage=0

. utils/parse_options.sh

for f in data/train_sup/utt2spk data/train/utt2spk \
  data/train_sup/text \
  data/lang_test_poco_sup100k/G.fst \
  data/lang_test_poco_sup100k_unk/G.fst
  ; do
  if [ ! -f $f ]; then
    echo "$0: Could not find $f"
    exit 1
  fi
done

if [ $stage -le 0 ]; then
  utils/subset_data_dir.sh --utt-list <(utils/filter_scp.pl --exclude data/train_sup/utt2spk data/train/utt2spk) data/train data/train_unsup100k
  utils/subset_data_dir.sh --speakers data/train_unsup100k 500000 data/train_unsup100k_500k
  utils/combine_data.sh data/semisup100k_500k data/train_sup data/train_unsup100k_500k
fi

###############################################################################
# Train seed chain system using 100 hours supervised data.
# Here we train i-vector extractor on only the supervised set.
###############################################################################

if [ $stage -le 1 ]; then
  local/semisup/chain/tuning/run_tdnn_lstm_1b.sh \
    --train-set train_sup \
    --ivector-train-set "" \
    --nnet3-affix "" --chain-affix "" \
    --tdnn-affix _1b --tree-affix bi_a \
    --gmm tri4a --exp-root $exp_root || exit 1
fi

###############################################################################
# Semi-supervised training using 100 hours supervised data and 
# 250 hours unsupervised data. We use i-vector extractor, tree, lattices 
# and seed chain system from the previous stage.
###############################################################################

if [ $stage -le 2 ]; then
  local/semisup/chain/tuning/run_tdnn_lstm_100k_semisupervised_1b.sh \
    --hidden-dim 1536 --cell-dim 1536 --projection-dim 384 \
    --supervised-set train_sup \
    --unsupervised-set train_unsup100k_500k \
    --lm-weights 3,1 \
    --sup-chain-dir $exp_root/chain/tdnn_lstm_1b_sp \
    --sup-lat-dir $exp_root/chain/tri4a_train_sup_unk_lats \
    --sup-tree-dir $exp_root/chain/tree_bi_a \
    --ivector-root-dir $exp_root/nnet3 \
    --chain-affix "" \
    --tdnn-affix _semisup100k_500k_1b \
    --exp-root $exp_root || exit 1
fi

###############################################################################
# Oracle system trained on combined 350 hours including both supervised and 
# unsupervised sets. We use i-vector extractor, tree, and GMM trained
# on only the supervised for fair comparison to semi-supervised experiments.
###############################################################################

if [ $stage -le 3 ]; then
  local/semisup/chain/tuning/run_tdnn_lstm_1b.sh \
    --hidden-dim 1536 --cell-dim 1536 --projection-dim 384 \
    --train-set semisup100k_500k \
    --nnet3-affix "" --chain-affix "" \
    --common-treedir $exp_root/chain/tree_bi_a \
    --tdnn-affix _oracle100k_500k_1b --nj 100 \
    --gmm tri4a --exp $exp_root \
    --stage 9 || exit 1
fi

