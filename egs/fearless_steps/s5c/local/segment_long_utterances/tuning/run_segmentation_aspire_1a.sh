#!/bin/bash

set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
data_whole=data/train_seg_1a_whole_hires
train_data_dir=data/train_1a_seg
src_dir=exp/chain/tdnn_lstm_1a
extractor=exp/nnet3/extractor
treedir=exp/chain/tree_bi_a
src_lang=data/lang
lang_test=data/lang_test

nj=80

segment_stage=-10
affix=_lu_wgt_1a

# training options
srand=0
remove_egs=true
# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

utils/lang/check_phones_compatible.sh $src_lang/phones.txt $src_dir/phones.txt || exit 1
utils/lang/check_phones_compatible.sh $src_lang/phones.txt $treedir/phones.txt || exit 1
utils/lang/check_phones_compatible.sh $src_lang/phones.txt $lang_test/phones.txt || exit 1

train_id=$(basename $train_data_dir)
#train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_id}

#if [ $stage -le 1 ]; then
#  steps/online/nnet2/extract_ivectors_online.sh \
#    --cmd "$train_cmd" --nj $nj \
#    ${train_data_dir}_hires $extractor \
#    exp/nnet3${nnet3_affix}/ivectors_${train_id} || exit 1
#
#  for d in dev; do 
#    steps/online/nnet2/extract_ivectors_online.sh \
#      --cmd "$train_cmd" --nj $nj \
#      data/${d}_hires $extractor \
#      exp/nnet3${nnet3_affix}/ivectors_${d} || exit 1
#  done
#fi

decode_opts="--extra-left-context 50 --extra-right-context 0 --extra-left-context-initial 0 --extra-right-context-final 0 --frames-per-chunk 150"

if [ $stage -le 2 ]; then
  workdir=${src_dir}/segmentation_${train_id}
  mkdir -p $workdir
  awk '{print $1" "$2}' ${train_data_dir}_hires/segments > \
    $workdir/utt2text

  steps/cleanup/segment_long_utterances_nnet3_b.sh \
    --extractor $extractor \
    $decode_opts \
    --nj $nj --num-jobs-align 16 \
    --cmd "$decode_cmd" --stage $segment_stage \
    ${src_dir} $lang_test \
    ${train_data_dir}_hires $data_whole/text $workdir/utt2text \
    ${train_data_dir}_segmented_1a2 $workdir
fi

if [ $stage -le 3 ]; then
  workdir=${src_dir}/cleanup_${train_id}_segmented_1a2_1a2
  steps/cleanup/clean_and_segment_data_nnet3.sh \
    --extractor $extractor \
    $decode_opts \
    --nj $nj \
    --cmd "$decode_cmd" \
    ${train_data_dir}_segmented_1a2 \
    $lang_test ${src_dir} $workdir ${train_data_dir}_segmented_1a2_cleaned_1a2 || exit 1
fi

exit 0

