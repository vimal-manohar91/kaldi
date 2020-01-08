#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

stage=1
nnet3_affix=_norvb
train_set=train
speed_perturb=true

set -e

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

mkdir -p exp/nnet3${nnet3_affix}

if $speed_perturb; then
  if [ $stage -le 1 ]; then
    # Although the nnet will be trained by high resolution data, we still have
    # to perturb the normal data to get the alignments.
    # _sp stands for speed-perturbed

    for data_dir in $train_set; do
      utils/data/perturb_data_dir_speed_3way.sh data/${data_dir} data/${data_dir}_sp
      utils/fix_data_dir.sh data/${data_dir}_sp

      mfccdir=mfcc
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
        data/${data_dir}_sp exp/make_mfcc/${data_dir}_sp $mfccdir || exit 1
      steps/compute_cmvn_stats.sh \
        data/${data_dir}_sp exp/make_mfcc/${data_dir}_sp $mfccdir || exit 1
      utils/fix_data_dir.sh data/${data_dir}_sp
    done

  fi
  train_set=${train_set}_sp
fi

if [ $stage -le 2 ]; then
  for data_dir in $train_set; do
    utils/copy_data_dir.sh data/${data_dir} data/${data_dir}_hires
    utils/data/perturb_data_dir_volume.sh data/${data_dir}_hires

    mfccdir=mfcc_hires
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 70 --mfcc-config conf/mfcc_hires.conf \
      data/${data_dir}_hires exp/make_hires/${data_dir} $mfccdir || exit 1
    steps/compute_cmvn_stats.sh \
      data/${data_dir}_hires exp/make_hires/${data_dir} $mfccdir || exit 1
    utils/fix_data_dir.sh data/${data_dir}_hires
  done

  utils/subset_data_dir.sh data/${train_set}_hires 100000 data/${train_set}_hires_100k
  utils/subset_data_dir.sh data/${train_set}_hires 30000 data/${train_set}_hires_30k
fi

if [ $stage -le 3 ]; then
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 30000 --subsample 2 \
    data/${train_set}_hires exp/nnet3${nnet3_affix}/pca_transform
fi

if [ $stage -le 4 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest
  # subset.  
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 400000 \
    data/${train_set}_hires_30k 512 exp/nnet3${nnet3_affix}/pca_transform \
    exp/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 5 ]; then
  # iVector extractors can in general be sensitive to the amount of data, but
  # this one has a fairly small dim (defaults to 100) so we don't use all of it,
  # we use just the 100k subset (about one sixteenth of the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/${train_set}_hires_100k exp/nnet3${nnet3_affix}/diag_ubm \
    exp/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 6 ]; then
  ivectordir=exp/nnet3${nnet3_affix}/ivectors_${train_set}
  if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then # this shows how you can split across multiple file-systems.
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/aspire/s5/$ivectordir/storage $ivectordir/storage
  fi

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 \
    data/${train_set}_hires data/${train_set}_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 60 \
    data/${train_set}_hires_max2 exp/nnet3${nnet3_affix}/extractor $ivectordir || exit 1;
fi
