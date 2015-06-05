#!/bin/bash

# This is to be run after run_nnet2.sh

. cmd.sh

use_preconditioning=true

stage=1
train_stage=-10
use_gpu=true
learning_rate=2e-4
srcdir=exp/nnet2_online/nnet_ms_a_100k
dir=exp/nnet2_online/nnet_ms_a_100k_smbr
num_jobs_nnet=4

datadir=data/train_hires_100k
train_id=
degs_dir=
denlats_dir=
ivector_dir=exp/nnet2_online/ivectors_train

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

[ -z "$train_id" ] && train_id=`basename $datadir`

dir=${dir}_lr${learning_rate}_nj${num_jobs_nnet}

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

if [ -z "$denlats_dir" ]; then
  denlats_dir=${srcdir}_denlats${train_id}
  if [ $stage -le 1 ]; then
    # the make_denlats job is always done on CPU not GPU, since in any case
    # the graph search and lattice determinization takes quite a bit of CPU.
    # note: it's the sub-split option that determinies how many jobs actually
    # run at one time.
    steps/nnet2/make_denlats.sh --cmd "$decode_cmd --mem 1G" \
        --nj $nj --sub-split 40 --num-threads 6 --parallel-opts "-pe smp 6" \
        --online-ivector-dir $ivector_dir \
        $datadir data/lang $srcdir ${srcdir}_denlats${train_id}
  fi
fi

if [ $stage -le 2 ]; then
  if $use_gpu; then use_gpu_opt=yes; else use_gpu_opt=no; fi
  steps/nnet2/align.sh  --cmd "$decode_cmd $gpu_opts" \
      --online-ivector-dir $ivector_dir \
      --use-gpu $use_gpu_opt \
      --nj $nj $datadir data/lang ${srcdir} ${srcdir}_ali${train_id}
fi

if [ $stage -le 3 ]; then
  if [ -z "$degs_dir" ]; then
    if [ $USER == dpovey ]; then # this shows how you can split across multiple file-systems.
      utils/create_split_dir.pl /export/b0{1,2,3,4}/dpovey/kaldi-online/egs/fisher_english/s5/$dir/degs $dir/degs/storage
    fi

    steps/nnet2/get_egs_discriminative2.sh --cmd "$train_cmd --max-jobs-run 10" \
      --criterion smbr --drop-frames true \
      --online-ivector-dir $ivector_dir \
      $datadir data/lang \
      ${srcdir}_ali${train_id} ${denlats_dir} $srcdir/final.mdl $dir/degs
  fi

fi

if [ $stage -le 4 ]; then
  [ -z "$degs_dir" ] && degs_dir=$dir/degs
  # decreasing the learning rate by a factor of 2, due to having so much data, 
  # and decreasing the number of epochs for the same reason.
  # the io-opts option is to have more get_egs (and similar) jobs running at a time,
  # since we're using 4 disks.
  steps/nnet2/train_discriminative2.sh --cmd "$decode_cmd" \
    --stage $train_stage \
    --learning-rate $learning_rate \
    --num-epochs 4 --adjust-priors true \
    --num-jobs-nnet $num_jobs_nnet  --num-threads $num_threads --parallel-opts "$gpu_opts" \
    $degs_dir $dir || exit 1
fi

if [ $stage -le 5 ]; then
  steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
    data/lang exp/nnet2_online/extractor "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 6 ]; then
  # we'll do the decoding as 'online' decoding by using the existing
  # _online directory but with extra models copied to it.
  for epoch in 1 2 3 4; do
    cp $dir/epoch${epoch}.mdl ${dir}_online/epoch${epoch}.mdl
  done

  for epoch in 2 3 4; do
    # do the actual online decoding with iVectors, carrying info forward from 
    # previous utterances of the same speaker.
    steps/online/nnet2/decode.sh --cmd "$decode_cmd" --nj 30 --iter epoch${epoch} \
       exp/tri4a/graph_100k data/dev ${dir}_online/decode_100k_dev_epoch${epoch} || exit 1;
    
    #steps/online/nnet2/decode.sh --cmd "$decode_cmd" --nj 30 --iter epoch${epoch} \
    #   exp/tri4a/graph data/dev ${dir}_online/decode_dev_epoch${epoch} || exit 1;
  done
fi

wait

# for results, see the end of run_nnet2.sh

