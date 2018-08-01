#!/bin/bash

nnet3_affix=
stage=1

orig_data_dir=data/ami_sdm1_train_sp_hires
student_data_dir=data/ami_sdm1_train_16kHz_sp_hires
student_mfcc_config=conf/mfcc_hires_16kHz.conf

test_sets="ami_sdm1_dev_16kHz ami_sdm1_eval_16kHz"

num_threads_ubm=16
nj=40

echo "$0 $@"  # Print the command line for logging

. ./path.sh
. ./cmd.sh

set -e -o pipefail -u

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  exit 1
fi

if [ $stage -le 1 ]; then
  if [ -f $student_data_dir/feats.scp ]; then
    echo "$0: $student_data_dir/feats.scp exists. Remove it and skip this stage."
    exit 1
  fi

  utils/copy_data_dir.sh $orig_data_dir $student_data_dir

  steps/make_mfcc.sh --mfcc-config $student_mfcc_config --cmd "$train_cmd" --nj $nj \
    $student_data_dir
  steps/compute_cmvn_stats.sh $student_data_dir
  utils/fix_data_dir.sh $student_data_dir
fi

if [ $stage -le 2 ]; then
  for dset in $test_sets; do
    utils/copy_data_dir.sh data/${dset} data/${dset}_hires
    steps/make_mfcc.sh --mfcc-config $student_mfcc_config --cmd "$train_cmd" --nj $nj \
      data/${dset}_hires
    steps/compute_cmvn_stats.sh data/${dset}_hires
    utils/fix_data_dir.sh data/${dset}_hires
  done
fi

if [ $stage -le 3 ]; then
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 30000 --subsample 2 \
    $student_data_dir exp/nnet3${nnet3_affix}/pca_transform
fi

if [ $stage -le 4 ]; then
  num_utts=$(cat $student_data_dir/utt2spk | wc -l)
  suffix=
  if [ $num_utts -gt 30000 ]; then
    utils/subset_data_dir.sh $student_data_dir 30000 ${student_data_dir}_30k
    suffix=_30k
  fi

  # To train a diagonal UBM we don't need very much data, so use the smallest
  # subset.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj $nj \
    --num-frames 400000 --num-threads $num_threads_ubm \
    ${student_data_dir}${suffix} 512 exp/nnet3${nnet3_affix}/pca_transform \
    exp/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 5 ]; then
  num_utts=$(cat $student_data_dir/utt2spk | wc -l)
  suffix=
  if [ $num_utts -gt 100000 ]; then
    utils/subset_data_dir.sh $student_data_dir 100000 ${student_data_dir}_100k
    suffix=_100k
  fi
  # iVector extractors can in general be sensitive to the amount of data, but
  # this one has a fairly small dim (defaults to 100) so we don't use all of it,
  # we use just the 100k subset (about one sixteenth of the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj $nj \
    ${student_data_dir}${suffix} exp/nnet3${nnet3_affix}/diag_ubm \
    exp/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 6 ]; then
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    $student_data_dir ${student_data_dir}_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${student_data_dir}_max2 exp/nnet3${nnet3_affix}/extractor \
    exp/nnet3${nnet3_affix}/ivectors_$(basename $student_data_dir)

  for dset in $test_sets; do 
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
      data/${dset}_hires exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/ivectors_$dset
  done
fi
