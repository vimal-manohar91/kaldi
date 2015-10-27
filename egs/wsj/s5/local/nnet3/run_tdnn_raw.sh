#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=8
train_stage=-10
skip_lda=true
dir=exp/nnet3/nnet_tdnn_a
init_lr=0.001
final_lr=0.0001
egs_dir=
momentum=0.0
max_change=1.0
ivector_dir=
add_layers_period=5
init_job=2
final_job=14
relu_dim=750
num_relu=
target_rms=0.2
b_dim=
b_layer=1
splice_dim="-4,-3,-2,-1,0,1,2,3,4  0  -2,2  0  -4,4 0"
num_epochs=12
pnorm_input_dim=2000
pnorm_output_dim=250
add_log_sum=false
#ivector_dir=exp/nnet3/ivectors_train_si284
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

if [ ! -z "$relu_dim" ];then 
  opts="--relu-dim $relu_dim"
fi
if [ ! -z "$num_relu" ];then
  opts="$opts --num-relu $num_relu"
fi
if [ ! -z "$egs_dir" ];then
  opts="$opts --egs-dir $egs_dir"
fi

if [ ! -z "$b_dim" ];then 
  opts="$opts --bottleneck-dim $b_dim --bottleneck-layer $b_layer"
fi

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_tdnn${skip_lda:+"_raw"}.sh --stage $train_stage \
    --num-epochs $num_epochs  --num-jobs-initial 2 --num-jobs-final 14 \
    --splice-indexes "$splice_dim" \
    --feat-type raw \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --io-opts "-tc 12" \
    --initial-effective-lrate $init_lr --final-effective-lrate $final_lr \
    --cmd "$decode_cmd" \
    --pnorm-input-dim $pnorm_input_dim \
    --pnorm-output-dim $pnorm_output_dim \
    --skip-lda $skip_lda \
    --momentum $momentum \
    --max-param-change $max_change \
    --add-layers-period $add_layers_period \
    --target-rms $target_rms $opts \
    --remove-egs false \
    --add-log-sum $add_log_sum \
    data/train_si284_hires data/lang exp/tri4b_ali_si284 $dir  || exit 1;
fi


if [ $stage -le 9 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
  for lm_suffix in tgpr bd_tgpr; do
    graph_dir=exp/tri4b/graph_${lm_suffix}
    # use already-built graphs.
    for year in eval92 dev93; do
      steps/nnet3/decode_raw.sh --nj 8 --cmd "$decode_cmd" --target-rms $target_rms \
         $graph_dir data/test_${year}_hires $dir/decode_${lm_suffix}_${year} || exit 1;
    done
  done
fi


exit 0;

# results:
grep WER exp/nnet3/nnet_tdnn_a/decode_{tgpr,bd_tgpr}_{eval92,dev93}/scoring_kaldi/best_wer
exp/nnet3/nnet_tdnn_a/decode_tgpr_eval92/scoring_kaldi/best_wer:%WER 6.03 [ 340 / 5643, 74 ins, 20 del, 246 sub ] exp/nnet3/nnet_tdnn_a/decode_tgpr_eval92/wer_13_1.0
exp/nnet3/nnet_tdnn_a/decode_tgpr_dev93/scoring_kaldi/best_wer:%WER 9.35 [ 770 / 8234, 162 ins, 84 del, 524 sub ] exp/nnet3/nnet_tdnn_a/decode_tgpr_dev93/wer_11_0.5
exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_eval92/scoring_kaldi/best_wer:%WER 3.81 [ 215 / 5643, 30 ins, 18 del, 167 sub ] exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_eval92/wer_10_1.0
exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_dev93/scoring_kaldi/best_wer:%WER 6.74 [ 555 / 8234, 69 ins, 72 del, 414 sub ] exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_dev93/wer_11_0.0
b03:s5:
