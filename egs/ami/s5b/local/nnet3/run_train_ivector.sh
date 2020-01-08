#!/bin/bash

set -e -o pipefail


# This script is called from local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh (and may eventually
# be called by more scripts).  It contains the common feature preparation and iVector-related parts
# of the script.  See those scripts for examples of usage.

stage=0
nj=30
train_data_dir=data/ihm/train_cleaned
exp_root=exp/ihm

num_threads_ubm=32
nnet3_affix=_cleaned     # affix for $exp_root/nnet3 directory to put iVector stuff in, so it
                         # becomes $exp_root/nnet3_cleaned or whatever.

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


for f in ${train_data_dir}/feats.scp; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done


if [ $stage -le 1 ]; then
  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 10000 --subsample 2 \
    $train_data_dir \
    $exp_root/nnet3${nnet3_affix}/pca_transform
fi

train_set=$(basename $train_data_dir)

if [ $stage -le 5 ]; then
  echo "$0: computing a subset of data to train the diagonal UBM."

  mkdir -p $exp_root/nnet3${nnet3_affix}/diag_ubm
  temp_data_root=$exp_root/nnet3${nnet3_affix}/diag_ubm

  # train a diagonal UBM using a subset of about a quarter of the data
  # we don't use the _comb data for this as there is no need for compatibility with
  # the alignments, and using the non-combined data is more efficient for I/O
  # (no messing about with piped commands).
  num_utts_total=$(wc -l <$train_data_dir/utt2spk)
  num_utts=$[$num_utts_total/4]
  utils/data/subset_data_dir.sh ${train_data_dir} \
      $num_utts ${temp_data_root}/${train_set}_subset

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
    --num-frames 700000 \
    --num-threads $num_threads_ubm \
    ${temp_data_root}/${train_set}_subset 512 \
    $exp_root/nnet3${nnet3_affix}/pca_transform $exp_root/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 6 ]; then
  # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data.  The script defaults to an iVector dimension of
  # 100.
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    ${train_data_dir} $exp_root/nnet3${nnet3_affix}/diag_ubm $exp_root/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 7 ]; then
  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data, the utterance list is the same.
  ivectordir=$exp_root/nnet3${nnet3_affix}/ivectors_${train_set}
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/ivectors/ami-$mic-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  fi
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  temp_data_root=${ivectordir}
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    ${train_data_dir} ${temp_data_root}/${train_set}_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${temp_data_root}/${train_set}_max2 \
    $exp_root/nnet3${nnet3_affix}/extractor $ivectordir
fi

exit 0;
