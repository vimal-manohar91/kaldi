#!/bin/bash

# This is to be run after run_nnet2.sh

. cmd.sh

use_preconditioning=true

stage=1
train_stage=-10
use_gpu=true
learning_rate=2e-4
srcdir=exp/nnet2_online/nnet_ms_a_100k
num_jobs_nnet="2 5"
learning_rate_scales="1.0 1.0"
criterion=smbr
criterion_unsup=esmbr
unsup_dir=unsup_100k_250k
src_models=
dir=
unsup_nj=64
num_epochs=4
skip_last_layer=true
single_nnet=true
deletion_penalty=0.0

ivectordir=
latdir=
degs_dir=
uegs_dir=

graph_src=exp/tri4a

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ -z "$dir" ]; then
  dir=${srcdir}_c${criterion}_${criterion_unsup}
else 
  dir=${dir}_c${criterion}_${criterion_unsup}
fi

if [ $unsup_dir != unsup_100k_250k ]; then
  dir=${dir}_${unsup_dir}
fi

dir=${dir}_modifylr_unsupscale_$(echo $learning_rate_scales | awk '{printf $2}')_lr${learning_rate}
dir=${dir}_nj$(echo $num_jobs_nnet | sed 's/ /_/g')

if ! $skip_last_layer; then
  dir=${dir}_noskip
fi
if ! $single_nnet; then
  dir=${dir}_singlennet
fi

if [ $deletion_penalty != "0.0" ]; then
  dir=${dir}_dp${deletion_penalty}
fi

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  gpu_opts="--gpu 1"
  train_parallel_opts="--gpu 1"
  num_threads=1
  # the _a is in case I want to change the parameters.
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  gpu_opts=""
  num_threads=16
  train_parallel_opts="--num-threads 16"
fi

set -e

nj=40

if [ -z "$degs_dir" ]; then
  degs_dir=$dir/degs

  if [ $stage -le 1 ]; then
    # the make_denlats job is always done on CPU not GPU, since in any case
    # the graph search and lattice determinization takes quite a bit of CPU.
    # note: it's the sub-split option that determinies how many jobs actually
    # run at one time.
    steps/nnet2/make_denlats.sh --cmd "$decode_cmd --mem 1G" \
      --nj $nj --sub-split 40 --num-threads 6 --parallel-opts "--num-threads 6" \
      --online-ivector-dir exp/nnet2_online/ivectors_train \
      data/train_hires_100k data/lang $srcdir ${srcdir}_denlats_100k
  fi

  if [ $stage -le 2 ]; then
    if $use_gpu; then use_gpu_opt=yes; else use_gpu_opt=no; fi
    steps/nnet2/align.sh  --cmd "$decode_cmd $gpu_opts" \
      --online-ivector-dir exp/nnet2_online/ivectors_train \
      --use-gpu $use_gpu_opt \
      --nj $nj data/train_hires_100k data/lang ${srcdir} ${srcdir}_ali_100k
  fi

  if [ $stage -le 3 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $degs_dir/storage ]; then
      utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english-$(date +'%d_%m_%H_%M')/$degs_dir $degs_dir/storage
    fi

    steps/nnet2/get_egs_discriminative2.sh --cmd "$train_cmd --max-jobs-run 10" \
      --criterion smbr --drop-frames true \
      --online-ivector-dir exp/nnet2_online/ivectors_train \
      data/train_hires_100k data/lang \
      ${srcdir}_ali_100k ${srcdir}_denlats_100k $srcdir/final.mdl $dir/degs
  fi
fi

if [ -z "$uegs_dir" ]; then
  uegs_dir=$dir/uegs

  if [ $stage -le 4 ]; then
    rm -rf data/${unsup_dir}_hires
    utils/copy_data_dir.sh data/${unsup_dir} data/${unsup_dir}_hires || exit 1
    steps/make_mfcc.sh --nj $unsup_nj \
      --mfcc-config ${srcdir}_online/conf/mfcc.conf \
      --cmd "$train_cmd" data/${unsup_dir}_hires \
      exp/make_hires/${unsup_dir} mfcc || exit 1

    steps/compute_cmvn_stats.sh data/${unsup_dir}_hires \
      exp/make_hires/${unsup_dir} mfcc || exit 1

    utils/fix_data_dir.sh data/${unsup_dir}_hires || exit 1
  fi

  if [ -z "$ivectordir" ]; then
    ivectordir=exp/nnet2_online/ivectors_${unsup_dir}
    if [ $stage -le 5 ]; then
      if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then # this shows how you can split across multiple file-systems.
        utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english-$(date +'%d_%m_%H_%M')/$ivectordir $ivectordir/storage
      fi

      steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $unsup_nj \
        data/${unsup_dir}_hires exp/nnet2_online/extractor $ivectordir || exit 1
    fi
  fi

  if [ -z "$latdir" ]; then
    latdir=${srcdir}/decode_100k_${unsup_dir}
    if [ $stage -le 6 ]; then
      steps/nnet2/decode.sh --cmd "$decode_cmd --mem 2G" --num-threads 6 \
        --nj $nj \
        --online-ivector-dir exp/nnet2_online/ivectors_${unsup_dir} \
        ${graph_src}/graph_100k data/${unsup_dir}_hires ${srcdir}/decode_100k_${unsup_dir}
    fi
  fi

  if [ $stage -le 7 ]; then
    local/best_path_weights.sh --cmd "$decode_cmd" --create-ali-dir true \
      data/${unsup_dir}_hires ${graph_src}/graph_100k $latdir \
      ${srcdir}/best_path_100k_${unsup_dir} || exit 1
  fi

  if [ $stage -le 8 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $uegs_dir/storage ]; then
      utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english-$(date +'%d_%m_%H_%M')/$uegs_dir $uegs_dir/storage
    fi

    steps/nnet2/get_uegs2.sh --cmd "$decode_cmd --max-jobs-run 5" \
      --online-ivector-dir $ivectordir \
      --alidir ${srcdir}/best_path_100k_${unsup_dir} \
      data/${unsup_dir}_hires data/lang \
      $latdir $srcdir/final.mdl $uegs_dir
  fi
fi

if [ $stage -le 9 ]; then
  # decreasing the learning rate by a factor of 2, due to having so much data, 
  # and decreasing the number of epochs for the same reason.
  # the io-opts option is to have more get_egs (and similar) jobs running at a time,
  # since we're using 4 disks.
  bash -x steps/nnet2/train_discriminative_semisupervised2.sh \
    --cmd "$decode_cmd" \
    --stage $train_stage \
    --learning-rate $learning_rate \
    --modify-learning-rates true \
    --separate-learning-rates true \
    --learning-rate-scales "$learning_rate_scales" \
    --cleanup false --deletion-penalty $deletion_penalty \
    --num-jobs-nnet "$num_jobs_nnet" --num-threads $num_threads --parallel-opts "$gpu_opts" \
    --criterion $criterion --drop-frames true \
    --criterion-unsup $criterion_unsup \
    --num-epochs $num_epochs --adjust-priors true \
    --skip-last-layer $skip_last_layer --src-models "$src_models" \
    --single-nnet $single_nnet \
    $degs_dir $uegs_dir $dir || exit 1
fi

if [ $stage -le 10 ]; then
  steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
    data/lang exp/nnet2_online/extractor $dir/0 ${dir}/0_online || exit 1;
fi

if [ $stage -le 11 ]; then
  # we'll do the decoding as 'online' decoding by using the existing
  # _online directory but with extra models copied to it.
  lang=0
  for epoch in `seq $[num_epochs-1] $num_epochs`; do
    cp $dir/$lang/epoch${epoch}.mdl ${dir}/${lang}_online/epoch${epoch}.mdl

    # do the actual online decoding with iVectors, carrying info forward from 
    # previous utterances of the same speaker.
    steps/online/nnet2/decode.sh --cmd "$decode_cmd" --nj 30 \
      --iter epoch${epoch} \
      exp/tri4a/graph_100k data/dev ${dir}/${lang}_online/decode_100k_dev_epoch${epoch} || exit 1;
  done
fi

wait

# for results, see the end of run_nnet2.sh


