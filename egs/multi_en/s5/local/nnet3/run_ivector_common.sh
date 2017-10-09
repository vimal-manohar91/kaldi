#!/bin/bash

###########################################################################################
# This script was copied from egs/fisher_swbd/s5/local/nnet3/run_ivector_common.sh
# The source commit was e69198c3dc5633f98eb88e1cdf20b2521a598f21
# Changes made:
#  - Modified paths to match multi_en naming conventions
###########################################################################################

. ./cmd.sh
set -e
stage=1
train_stage=-10
speed_perturb=true
multi=multi_a
train_set=tdnn

. ./path.sh
. ./utils/parse_options.sh

# perturbed data preparation
if [ "$speed_perturb" == "true" ]; then
  if [ $stage -le 1 ]; then
    #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
    # _sp stands for speed-perturbed

    for datadir in $multi/$train_set; do
      utils/data/perturb_data_dir_speed_3way.sh data/${datadir} data/${datadir}_sp
      steps/make_mfcc.sh --nj 70 --cmd "$train_cmd" \
        data/${datadir}_sp || exit 1
      steps/compute_cmvn_stats.sh data/${datadir}_sp || exit 1
      utils/fix_data_dir.sh data/${datadir}_sp || exit 1
    done
  fi
  train_set=${train_set}_sp
fi

if [ $stage -le 3 ]; then
  for dataset in $multi/$train_set; do
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires

    # scale the waveforms, this is useful as we don't use CMVN
    data_dir=data/${dataset}_hires
    utils/data/perturb_data_dir_volume.sh $data_dir

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd" \
      data/${dataset}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${dataset}_hires || exit 1;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}_hires || exit 1;
  done
  
  utils/subset_data_dir.sh --speakers data/$multi/${train_set}_hires 100000 \
    data/$multi/${train_set}_100k_hires
  utils/data/remove_dup_utts.sh 200 data/$multi/${train_set}_100k_hires \
    data/$multi/${train_set}_100k_nodup_hires
fi

if [ $stage -le 4 ]; then
  for dataset in eval2000 rt03; do
    # Create MFCCs for the eval set
    utils/copy_data_dir.sh data/$dataset/test data/${dataset}_hires/test
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 --mfcc-config conf/mfcc_hires.conf \
        data/${dataset}_hires/test || exit 1;
    steps/compute_cmvn_stats.sh data/${dataset}_hires/test || exit 1;
    utils/fix_data_dir.sh data/${dataset}_hires/test  # remove segments with problems
  done

  # Take the first 30k utterances, which will be used for the diagubm training
  utils/subset_data_dir.sh --first data/$multi/${train_set}_hires 30000 data/$multi/${train_set}_30k_hires
  utils/data/remove_dup_utts.sh 200 data/$multi/${train_set}_30k_hires data/$multi/${train_set}_30k_nodup_hires
fi

# ivector extractor training
if [ $stage -le 5 ]; then
  steps/online/nnet2/get_pca_transform.sh --max-utts 10000 \
    data/$multi/${train_set}_30k_nodup_hires exp/$multi/nnet3${nnet3_affix}/pca
fi

if [ $stage -le 6 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 200000 \
    data/$multi/${train_set}_30k_nodup_hires 512 exp/$multi/nnet3${nnet3_affix}/pca \
    exp/$multi/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 7 ]; then
  # iVector extractors can be sensitive to the amount of data, but this one has a
  # fairly small dim (defaults to 100) so we don't use all of it, we use just the
  # 100k subset (just under half the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/$multi/${train_set}_100k_nodup_hires exp/$multi/nnet3${nnet3_affix}/diag_ubm exp/$multi/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 8 ]; then
  # We extract iVectors on all the train_nodup data, which will be what we
  # train the system on.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/$multi/${train_set}_hires exp/$multi/nnet3${nnet3_affix}/${train_set}_max2_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    exp/$multi/nnet3${nnet3_affix}/${train_set}_max2_hires exp/$multi/nnet3${nnet3_affix}/extractor \
    exp/$multi/nnet3${nnet3_affix}/ivectors_$train_set || exit 1;
  
  for data_set in eval2000 rt03; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
      data/${data_set}_hires/test exp/$multi/nnet3/extractor exp/$multi/nnet3/ivectors_$data_set || exit 1;
  done
fi

exit 0;
