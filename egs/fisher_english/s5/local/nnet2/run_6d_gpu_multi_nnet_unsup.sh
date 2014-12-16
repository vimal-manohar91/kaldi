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
degs_dir=
degs_unsup_dir=
nj=30
learning_rate_scales="1.0 1.0"
learning_rate=9e-5
criterion=smbr
dir=exp/nnet_6d_gpu_multi_nnet_unsup
set -e # exit on error.

. ./cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
. utils/parse_options.sh

dir=${dir}_unsupscale_$(echo $learning_rate_scales | awk '{printf $1}')_lr${learning_rate}
best_path_dir=$srcdir/best_path_100k_unsup_100k_250k

if [ -z "$degs_dir" ]; then
  degs_dir=$dir/degs

  if [ $stage -le 1 ]; then 
    steps/nnet2/align.sh  --cmd "$decode_cmd $gpu_opts" \
      --use-gpu yes \
      --transform-dir exp/tri4a \
      --nj $nj data/train_100k data/lang ${srcdir} ${srcdir}_ali_100k
  fi

  if [ $stage -le 2 ]; then
    steps/nnet2/make_denlats.sh --cmd "$decode_cmd --mem 1G" \
      --nj $nj --sub-split 20 --num-threads 6 --parallel-opts "-pe smp 6" \
      --transform-dir exp/tri4a \
      data/train_100k data/lang $srcdir ${srcdir}_denlats_100k
  fi

  if [ $stage -le 3 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $degs_dir/storage ]; then
      utils/create_split_dir.pl /export/b0{1,2,3,4}/kaldi-data/egs/fisher_english-$(date +'%d_%m_%H_%M')/$degs_dir $degs_dir/storage
    fi

    steps/nnet2/get_egs_discriminative2.sh --cmd "$decode_cmd --max-jobs-run 5" \
      --criterion $criterion --drop-frames true \
      data/train_100k data/lang \
      ${srcdir}_ali_100k ${srcdir}_denlats_100k $srcdir/final.mdl $degs_dir
  fi
fi

if [ -z "$degs_unsup_dir" ]; then
  degs_unsup_dir=$dir/degs_unsup

  if [ $stage -le 4 ]; then
    local/best_path_weights.sh --create-ali-dir true --cmd "$decode_cmd" \
      data/unsup_100k_250k data/lang_100k_test \
      ${srcdir}/decode_100k_unsup_100k_250k $best_path_dir
  fi

  if [ $stage -le 5 ]; then
    steps/nnet2/make_denlats.sh --cmd "$decode_cmd --mem 1G" \
      --nj $nj --sub-split 20 --num-threads 6 --parallel-opts "-pe smp 6" \
      --transform-dir exp/tri4a/decode_100k_unsup_100k_250k \
      data/semisup_100k_250k data/lang \
      $srcdir ${srcdir}/denlats_100k_semisup_100k_250k
  fi

  if [ $stage -le 6 ]; then

    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $degs_unsup_dir/storage ]; then
      utils/create_split_dir.pl /export/b0{1,2,3,4}/kaldi-data/egs/fisher_english-$(date +'%d_%m_%H_%M')/$degs_unsup_dir $degs_unsup_dir/storage
    fi

    steps/nnet2/get_egs_discriminative2.sh --cmd "$decode_cmd --max-jobs-run 5" \
      --criterion $criterion --drop-frames true \
      data/semisup_100k_250k data/lang \
      $best_path_dir ${srcdir}/denlats_100k_semisup_100k_250k $srcdir/final.mdl $degs_unsup_dir
  fi
fi

if [ $stage -le 7 ]; then
  steps/nnet2/train_discriminative_multilang2.sh --cmd "$decode_cmd --gpu 1" \
    --stage $train_stage \
    --learning-rate $learning_rate --num-jobs-nnet "4 4" \
    --criterion $criterion --drop-frames true \
    --learning-rate-scales $learning_rate_scales \
    --cleanup false --remove-egs false \
    --num-epochs 4 --num-threads 1 \
    $degs_dir $degs_unsup_dir $dir
fi

if [ $stage -le 8  ]; then
  steps/nnet2/decode.sh --cmd "$decode_cmd" --num-threads 6 --mem $decode_mem \
    --nj 25 --config conf/decode.config \
    --transform-dir exp/tri4a/decode_100k_dev \
    exp/tri4a/graph_100k data/dev $dir/1/decode_100k_dev
fi
