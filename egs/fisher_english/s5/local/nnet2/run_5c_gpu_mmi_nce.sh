#!/bin/bash


# this (local/nnet2/run_5c_gpu_nce.sh) trains a p-norm neural network 
# in an unsupervised way using NCE criterion on top of nnet_5c_gpu

src_dir=exp/nnet5c_gpu_nce
dir=
train_stage=-10
learning_rate=9e-5
num_epochs=1
uegs_dir=""
degs_dir=""
create_degs_dir=true
create_uegs_dir=true
egs_dir=""

. ./cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF


. utils/parse_options.sh
parallel_opts="-q g.q -l gpu=1"  # This is suitable for the CLSP network, you'll likely have to change it.

if [ -z "$dir" ]; then
  dir=`echo $src_dir | sed "s:$src_dir:${src_dir}_mmi_nce:"`
fi

dir=${dir}_lr$(echo $learning_rate | sed 's/-/m/')
ali_dir=${src_dir}_ali_100k
denlats_dir=${src_dir}_denlats_100k

if [ ! -f $ali_dir/.done ]; then
  steps/nnet2/align.sh --cmd "$train_cmd" --nj 32 \
    --transform-dir exp/tri4a_ali_100k \
    data/train_100k data/lang $src_dir ${ali_dir} || exit 1
  touch $ali_dir/.done
fi

if [ ! -f $denlats_dir/.done ]; then

  steps/nnet2/make_denlats.sh --cmd "$decode_cmd -l mem_free=1G,ram_free=1G" \
    --parallel-opts "-pe smp 6" --num-threads 6 --nj 32 --sub-split 20 \
    --transform-dir exp/tri4a_ali_100k data/train_100k data/lang \
    $src_dir $denlats_dir || exit 1
  touch $denlats_dir/.done
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

degs_dir_orig=$degs_dir
uegs_dir_orig=$uegs_dir

( 
if [ ! -f $dir/.done ]; then 
  $create_degs_dir && degs_dir=""
  $create_uegs_dir && uegs_dir=""

  if [[ `hostname -f` == *.clsp.jhu.edu ]]; then
    # spread the egs over various machines. 
    [ -z "$uegs_dir" ] && utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english_s5/exp/nnet5c_gpu_nce/uegs $dir/uegs/storage
    [ -z "$degs_dir" ] && utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english_s5/exp/nnet5c_gpu_nce/degs $dir/degs/storage 
  fi

  steps/nnet2/train_discriminative_semisupervised.sh \
    --stage $train_stage --cmd "$decode_cmd" \
    --learning-rate $learning_rate \
    --modify-learning-rates true \
    --last-layer-factor 0.1 \
    --num-epochs $num_epochs \
    --cleanup false \
    --io-opts "-tc 10" \
    --transform-dir-unsup exp/tri4a/decode_100k_unsup_100k_250k \
    --transform-dir-sup exp/tri4a_ali_100k \
    --degs-dir "$degs_dir" --uegs-dir "$uegs_dir" \
    --num-jobs-nnet-unsup 4 --num-threads 1 \
    --num-jobs-nnet-sup 4 \
    --parallel-opts "$parallel_opts" \
    --cmd "$decode_cmd" \
    data/train_100k data/unsup_100k_250k data/lang ${ali_dir} $denlats_dir \
    $src_dir/decode_100k_unsup_100k_250k $src_dir $dir || exit 1;

  $create_degs_dir && rm -rf $degs_dir_orig && mv $degs_dir $degs_dir_orig
  $create_uegs_dir && rm -rf $uegs_dir_orig && mv $uegs_dir $uegs_dir_orig

  degs_dir=$degs_dir_orig
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

if [[ `hostname -f` == *.clsp.jhu.edu ]]; then
  [ -z "$egs_dir" ] && utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english_s5/exp/nnet5c_gpu_nce_update/egs ${dir}_update/egs/storage
fi

if [ ! -f ${dir}_update/.done ]; then
  steps/nnet2/update_nnet.sh \
    --cmd "$train_cmd" \
    --parallel-opts "-q g.q -l gpu=1" --num-threads 1 --num-jobs-nnet 4 \
    --num-epochs 1 --num-iters-final 4 \
    --learning-rates "0:0:0:0:0.00008" \
    --egs-dir "$egs_dir" --cleanup false \
    --transform-dir exp/tri4a_ali_100k data/train_100k data/lang \
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


