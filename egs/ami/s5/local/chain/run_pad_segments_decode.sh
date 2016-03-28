#!/bin/bash

set -e 
set -o pipefail

mic=ihm
dir=exp/chain/tdnn_ami4_sp

. cmd.sh
. path.sh

. utils/parse_options.sh

pad_length_left=0.02
pad_length_right=0.04

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$dir/graph_${LM}

for decode_set in dev eval; do
  suffix=pad${pad_length_left}_${pad_length_right}
  num_jobs=`cat data/$mic/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`

  steps/cleanup/pad_data_dir.sh --nj $num_jobs --cmd "$train_cmd" \
    data/$mic/${decode_set}_hires \
    exp/$mic/pad_segments_${decode_set} data/$mic/${decode_set}_${suffix}_hires

  decode_set=${decode_set}_$suffix

  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd" --nj $num_jobs \
    data/$mic/${decode_set}_hires exp/$mic/make_hires mfcc_hires
  steps/compute_cmvn_stats.sh \
    data/$mic/${decode_set}_hires exp/$mic/make_hires mfcc_hires
  utils/fix_data_dir.sh data/$mic/${decode_set}_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $num_jobs \
    data/$mic/${decode_set}_hires \
    exp/$mic/nnet3/extractor \
    exp/$mic/nnet3/ivectors_${decode_set} || exit 1;

  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --nj $num_jobs --cmd "$decode_cmd" \
    --online-ivector-dir exp/$mic/nnet3/ivectors_${decode_set} \
    $graph_dir data/$mic/${decode_set}_hires $dir/decode_${decode_set} || exit 1;
done
