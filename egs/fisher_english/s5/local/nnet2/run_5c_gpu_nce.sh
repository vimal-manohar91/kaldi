#!/bin/bash


# this (local/nnet2/run_5c_gpu_nce.sh) trains a p-norm neural network 
# in an unsupervised way using NCE criterion on top of nnet_5c_gpu

src_dir=exp/nnet5c_gpu_nce
dir=
train_stage=-10
learning_rate=9e-6
num_epochs=1
uegs_dir=""
egs_dir=""
create_egs_dir=true
create_uegs_dir=true

. ./cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF


. utils/parse_options.sh
parallel_opts="-l gpu=1"  # This is suitable for the CLSP network, you'll likely have to change it.

if [ -z "$dir" ]; then
  dir=`echo $src_dir | sed "s:$src_dir:${src_dir}_nce:"`
fi

dir=${dir}_lr$(echo $learning_rate | sed 's/-/m/')
ali_dir=${src_dir}_ali_100k

if [ ! -f $ali_dir/.done ]; then
  steps/nnet2/align.sh --cmd "$train_cmd" --nj 32 \
    --transform-dir exp/tri4a \
    data/train_100k data/lang $src_dir ${ali_dir} || exit 1
fi

decode=$src_dir/decode_100k_unsup_100k_250k
if [ ! -f $decode/.done ]; then
  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 32 \
    --parallel-opts "-pe smp 6" --num-threads 6 \
    --config conf/decode.config \
    --transform-dir exp/tri4a/decode_100k_unsup_100k_250k \
    exp/tri4a/graph_100k data/unsup_100k_250k $decode || exit 1
  touch $decode/.done
fi

egs_dir_orig=$egs_dir
uegs_dir_orig=$uegs_dir

( 
if [ ! -f $dir/.done ]; then 
  $create_egs_dir && egs_dir=""
  $create_uegs_dir && uegs_dir=""

  if [[ `hostname -f` == *.clsp.jhu.edu ]]; then
    # spread the egs over various machines. 
    [ -z "$egs_dir" ] && utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english_s5/exp/nnet5c_gpu_nce/uegs $dir/uegs/storage
    [ -z "$uegs_dir" ] && utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english_s5/exp/nnet5c_gpu_nce/egs $dir/egs/storage 
  fi

  steps/nnet2/train_discriminative_unsupervised.sh \
    --stage $train_stage --cmd "$decode_cmd" \
    --learning-rate $learning_rate \
    --modify-learning-rates true \
    --last-layer-factor 0.1 \
    --num-epochs $num_epochs \
    --cleanup false \
    --io-opts "-tc 10" \
    --transform-dir-unsup exp/tri4a/decode_100k_unsup_100k_250k \
    --transform-dir-sup exp/tri4a \
    --egs-dir "$egs_dir" --uegs-dir "$uegs_dir" \
    --num-jobs-nnet 8 --num-threads 1 \
    --parallel-opts "$parallel_opts" \
    --cmd "$decode_cmd" \
    --last-layer-factor 0.1 \
    data/train_100k data/unsup_100k_250k data/lang ${ali_dir} \
    $src_dir/decode_100k_unsup_100k_250k $src_dir $dir || exit 1;

  $create_egs_dir && mv $egs_dir $egs_dir_orig
  $create_uegs_dir && mv $uegs_dir $uegs_dir_orig

  egs_dir=$egs_dir_orig
  uegs_dir=$uegs_dir_orig

  touch $dir/.done 
fi

(
decode=$dir/decode_100k_dev_epoch$num_epochs
if [ ! -f $decode/.done ]; then
  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 25 \
    --parallel-opts "-pe smp 6" --num-threads 6 \
    --config conf/decode.config --transform-dir exp/tri4a/decode_100k_dev \
    exp/tri4a/graph_100k data/dev $decode
  touch $decode/.done
fi
) &

[ -z "$egs_dir" ] && egs_dir=$dir/egs

if [ ! -f ${dir}_update/.done ]; then
  steps/nnet2/update_nnet.sh \
    --cmd "$train_cmd" \
    --parallel-opts "-l gpu=1" --num-threads 1 --num-jobs-nnet 4 \
    --num-epochs 1 --num-iters-final 4 \
    --learning-rates "0:0:0:0:0.00008" \
    --egs-dir "$egs_dir" \
    --transform-dir exp/tri4a data/train_100k data/lang \
    $ali_dir $dir ${dir}_update || exit 1
  touch ${dir}_update/.done
fi

(
decode=${dir}_update/decode_100k_dev_epoch$num_epochs
if [ ! -f $decode/.done ]; then
  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 25 \
    --parallel-opts "-pe smp 6" --num-threads 6 \
    --config conf/decode.config --transform-dir exp/tri4a/decode_100k_dev \
    exp/tri4a/graph_100k data/dev $decode
  touch $decode/.done
fi
) &


)

