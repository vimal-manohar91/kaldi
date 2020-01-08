#!/bin/bash

srcdir=exp/chain_cleaned/tdnn_lstm_bab9_sp
treedir=exp/chain_cleaned/tree
stage=-2
train_stage=-10

. utils/parse_options.sh

if [ ! -f data/train_flp/utt2spk ]; then
  echo "$0: Could not find data/train_flp/utt2spk. Copy the directory from the FullLP system directory"
  exit 1
fi

if [ $stage -le -2 ]; then
  utils/subset_data_dir.sh --spk-list <(utils/filter_scp.pl --exclude data/train/spk2utt data/train_flp/spk2utt) \
    data/train_flp data/unsup.pem
fi

local/chain/tuning/run_tdnn_lstm_semisup_1a.sh \
  --srcdir $srcdir \
  --treedir $treedir \
  --unsupervised-set unsup.pem \
  --nnet3-affix _cleaned_unsup.pem \
  --chain-affix _cleaned_unsup.pem \
  --stage $stage --train-stage $train_stage
