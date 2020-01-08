#!/bin/bash

nj=30
nnet3_affix=
mic=ihm

num_threads_ubm=32
max_jobs_run=10

. utils/parse_options.sh
. cmd.sh

if [ $# -ne 1 ]; then
  echo "Usage: $0 <data-dir>"
  exit 1
fi

data_dir=exp/$mic/nnet3${nnet3_affix}/combined_data_dir
utils/combine_data.sh $data_dir $*

if [ $stage -le 1 ]; then
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 30000 --subsample 2 \
    $data_dir \
    exp/$mic/nnet3${nnet3_affix}/pca_transform
fi


if [ $stage -le 2 ]; then
  echo "$0: computing a subset of data to train the diagonal UBM."

  mkdir -p exp/$mic/nnet3${nnet3_affix}/diag_ubm
  temp_data_root=exp/$mic/nnet3${nnet3_affix}/diag_ubm

  # train a diagonal UBM using a subset of about a quarter of the data,
  # and using the non-combined data is more efficient for I/O
  # (no messing about with piped commands).
  num_utts_total=$(wc -l <${data_dir}/utt2spk)
  num_utts=$[$num_utts_total/4]
  utils/data/subset_data_dir.sh ${data_dir} \
    $num_utts ${temp_data_root}/$(basename $data_dir)_subset

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
    --num-frames 700000 \
    --num-threads $num_threads_ubm \
    ${temp_data_root}/$(basename $data_dir)_subset 512 \
    exp/$mic/nnet3${nnet3_affix}/pca_transform exp/$mic/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 3 ]; then
  # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data.  The script defaults to an iVector dimension of
  # 100.
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    ${data_dir} exp/$mic/nnet3${nnet3_affix}/diag_ubm exp/$mic/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 4 ]; then
  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data, the utterance list is the same.

  for data in $*; do
    ivectordir=exp/$mic/nnet3${nnet3_affix}/ivectors_$(basename $data)
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
      utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$mic-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
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
      ${data} ${temp_data_root}/$(basename $data)_max2

    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
      ${temp_data_root}/$(basename $data)_max2 \
      exp/$mic/nnet3${nnet3_affix}/extractor $ivectordir
  done

  # Also extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp) or small-segment concatenation (comb).
  for data in dev eval; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "$nj" \
      data/${mic}/${data}_hires exp/$mic/nnet3${nnet3_affix}/extractor \
      exp/$mic/nnet3${nnet3_affix}/ivectors_${data}_hires
  done

exit 0;

