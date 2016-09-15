#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=8
train_stage=-10
num_epochs=8
splice_indexes="-2,-1,0,1,2 0 0 -2,0,2 -2,2 0 -4,4 0"
conv_self_repair_scale=0.00001
dir=exp/nnet3/nnet_tdnn_raw
egs_dir=exp/nnet3/nnet_tdnn_raw/egs
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

local/nnet3/run_ivector_common.sh --stage $stage || exit 1;

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_tdnn.sh --stage $train_stage \
    --use-raw-wave-feat true --max-input-shift 0.2 --remove-egs false --egs-dir $egs_dir \
    --num-epochs $num_epochs --num-jobs-initial 2 --num-jobs-final 8 \
    --splice-indexes "$splice_indexes" \
    --conv-opts "--conv-filter-dim 490 --conv-num-filters 40 --conv-filter-step 10 --conv-jesus-stddev-scale 1.0 --pnorm-block-dim 1 --conv-self-repair-scale $conv_self_repair_scale --use-shared-block true" \
    --jesus-opts "--jesus-forward-input-dim 750  --jesus-forward-output-dim 1280 --jesus-hidden-dim 12000 --jesus-stddev-scale 0.03 --final-layer-learning-rate-factor 0.25" \
    --feat-type raw \
    --online-ivector-dir exp/nnet3/ivectors_train_si284 \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate 0.0017 --final-effective-lrate 0.00017 \
    --cmd "$decode_cmd" \
    data/train_si284_hires data/lang exp/tri4b_ali_si284 $dir  || exit 1;
fi


if [ $stage -le 9 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
  for lm_suffix in tgpr bd_tgpr; do
    graph_dir=exp/tri4b/graph_${lm_suffix}
    # use already-built graphs.
    for year in eval92 dev93; do
      steps/nnet3/decode_raw.sh --nj 8 --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet3/ivectors_test_$year \
         $graph_dir data/test_${year}_hires $dir/decode_${lm_suffix}_${year} || exit 1;
    done
  done
fi


exit 0;

# results:
#grep WER exp/nnet3/nnet_tdnn_a/decode_{tgpr,bd_tgpr}_{eval92,dev93}/scoring_kaldi/best_wer
#exp/nnet3/nnet_tdnn_a/decode_tgpr_eval92/scoring_kaldi/best_wer:%WER 6.03 [ 340 / 5643, 74 ins, 20 del, 246 sub ] exp/nnet3/nnet_tdnn_a/decode_tgpr_eval92/wer_13_1.0
#exp/nnet3/nnet_tdnn_a/decode_tgpr_dev93/scoring_kaldi/best_wer:%WER 9.35 [ 770 / 8234, 162 ins, 84 del, 524 sub ] exp/nnet3/nnet_tdnn_a/decode_tgpr_dev93/wer_11_0.5
#exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_eval92/scoring_kaldi/best_wer:%WER 3.81 [ 215 / 5643, 30 ins, 18 del, 167 sub ] exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_eval92/wer_10_1.0
#exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_dev93/scoring_kaldi/best_wer:%WER 6.74 [ 555 / 8234, 69 ins, 72 del, 414 sub ] exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_dev93/wer_11_0.0
