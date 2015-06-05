#!/bin/bash

# this script is discriminative training after multi-language training (as
# run_nnet2_gale_combined_disc1.sh), but the discriminative training is
# multi-language too. 
# some of the stages are the same as run_nnet2_gale_combined_disc1.sh,
# and we didn't repeat them (we used the --stage option, it defaults to 4).

stage=0
gpu_opts="--gpu 1"
train_stage=-100
srcdir=exp/nnet5c_gpu_i3000_o300_n4
egs_dir=
uegs_dir=
nj=30
num_jobs_nnet="4 4"
learning_rate_scales="1.0 1.0"
learning_rate=9e-5
separate_learning_rates=false
skip_last_layer=true
num_epochs=4
dir=exp/nnet_6d_gpu_multi_nnet_more_nce_ml
set -e # exit on error.

. ./cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
. utils/parse_options.sh

dir=${dir}_modifylr_supscale_$(echo $learning_rate_scales | awk '{printf $2}')_lr${learning_rate}
if $separate_learning_rates; then
  dir=${dir}_separatelr
fi

if ! $skip_last_layer; then
  dir=${dir}_noskip
fi

if [ -z "$egs_dir" ]; then
  exit 1
  egs_dir=$dir/egs

  if [ $stage -le 2 ]; then 
    steps/nnet2/align.sh  --cmd "$decode_cmd $gpu_opts" \
      --use-gpu yes \
      --transform-dir exp/tri4a_ali_100k \
      --nj $nj data/train_100k data/lang ${srcdir} ${srcdir}_ali_100k
  fi

  if [ $stage -le 3 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $egs_dir/storage ]; then
      utils/create_split_dir.pl /export/b0{1,2,3,4}/kaldi-data/egs/fisher_english-$(date +'%d_%m_%H_%M')/$egs_dir $egs_dir/storage
    fi
    steps/nnet2/get_egs2.sh $egs_opts \
      --transform-dir exp/tri4a_ali_100k \
      --cmd "$train_cmd" --io-opts "--max-jobs-run 10" \
      data/train_100k \
      ${srcdir}_ali_100k $egs_dir
  fi
fi

if [ -z "$uegs_dir" ]; then
  uegs_dir=$dir/uegs

  if [ $stage -le 5 ]; then
    steps/nnet2/decode.sh --cmd "$decode_cmd --mem 2G --num-threads 6" \
      --nj $nj --sub-split 20 --num-threads 6 \
      --transform-dir exp/tri4a/decode_100k_unsup_100k_250k \
      data/unsup_100k_250k data/lang_100k_test \
      $srcdir ${srcdir}/decode_100k_unsup_100k_250k
  fi

  if [ $stage -le 6 ]; then

    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $uegs_dir/storage ]; then
      utils/create_split_dir.pl /export/b0{1,2,3,4}/kaldi-data/egs/fisher_english-$(date +'%d_%m_%H_%M')/$uegs_dir $uegs_dir/storage
    fi

    steps/nnet2/get_uegs2.sh --cmd "$decode_cmd --max-jobs-run 5" \
      --transform-dir exp/tri4a/decode_100k_unsup_100k_250k \
      data/unsup_100k_250k data/lang \
      ${srcdir}/decode_100k_unsup_100k_250k $srcdir/final.mdl $uegs_dir
  fi
fi

if [ $stage -le 7 ]; then
  steps/nnet2/train_more_multinnet2.sh \
    --cmd "$decode_cmd --gpu 1" \
    --stage $train_stage \
    --learning-rate $learning_rate \
    --modify-learning-rates true \
    --separate-learning-rates $separate_learning_rates \
    --learning-rate-scales "$learning_rate_scales" \
    --last-layer-factor 0.1 \
    --num-epochs $num_epochs \
    --cleanup false \
    --num-jobs-nnet "$num_jobs_nnet" --num-threads 1 \
    --objectives "nce ml" --drop-frames true --minibatch-size 512 \
    --src-models "$uegs_dir/final.mdl $uegs_dir/final.mdl" \
    $uegs_dir $egs_dir $dir
fi

if [ $stage -le 8  ]; then
  for lang in 0 1; do
    for epoch in `seq 1 $num_epochs`; do
      (
      steps/nnet2/decode.sh --cmd "$decode_cmd" --num-threads 6 --mem $decode_mem \
        --nj 25 --config conf/decode.config \
        --transform-dir exp/tri4a/decode_100k_dev \
        --iter epoch$epoch \
        exp/tri4a/graph_100k data/dev $dir/$lang/decode_100k_dev_epoch$epoch 
      ) &
    done
  done
fi


