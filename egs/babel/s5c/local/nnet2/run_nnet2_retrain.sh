#!/bin/bash

stage=-1
set -e 
set -o pipefail
set -u

train_stage=-100
use_gpu=true
srcdir=exp/multilang/tri6_nnet/0
lang=../kaldi/egs/SBS-mul-exp/data/lang/
alidir=../kaldi/egs/SBS-mul-exp/exp/tri3b
transform_dir=../kaldi/egs/SBS-mul-exp/exp/tri3b
data_dir=../kaldi/egs/SBS-mul-exp/data/train
trainfeats=exp/multilang/tri6_nnet_SBS-mul-noSW/train_activations
graph_dir=../kaldi/egs/SBS-mul-exp/exp/tri3b/graph
test_dir=../kaldi/egs/SBS-mul-exp/data/eval
test_transform_dir=../kaldi/egs/SBS-mul-exp/exp/tri3b/decode_eval
dir=exp/multilang/tri6_nnet_SBS-mul-noSW
nj=20
decode_nj=5
iter=final

learning_rate_opts="--initial-learning-rate 0.002 --final-learning-rate 0.0002" 
num_epochs=20
num_epochs_retrain=20

debug_mode=false
do_decode=false
skip_scoring=false

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=512
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="--num-threads $num_threads" 
fi

if [ $stage -le 0 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir//egs/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/babel-$date/s5c/$dir/egs/storage ${dir}/egs/storage
  fi

  steps/nnet2/get_egs2.sh \
    --io-opts "--max-jobs-run 10" --cmd "$train_cmd" \
    --transform-dir $transform_dir \
    --left-context 14 --right-context 10 \
    $data_dir $alidir $dir/egs
fi

if [ $stage -le 1 ]; then
  nnet-am-copy --learning-rates=0:0:0:0.008 $srcdir/final.mdl - | nnet-am-reinitialize - $alidir/final.mdl $srcdir/pre_init.mdl 

  steps/nnet2/train_more2.sh --learning-rate-factor 0.9 --cmd "$train_cmd" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-jobs-nnet 4 \
    --num-epochs $num_epochs \
     ${srcdir}/pre_init.mdl $dir/egs $dir
fi

graph_id=${graph_dir##*graph}
if $do_decode; then
  if [ $stage -le 3 ]; then
    steps/nnet2/decode.sh --config conf/decode.config --skip-scoring $skip_scoring \
      --cmd "$decode_cmd" --nj $decode_nj --transform-dir "$test_transform_dir" \
      $graph_dir $test_dir ${dir}/decode${graph_id}_`basename $test_dir`
  fi
fi

if $debug_mode; then
  echo "Exiting because --debug-mode true" && exit 0
fi

if [ $stage -le 6 ]; then

  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${dir}_combined/egs/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/babel-$date/s5c/${dir}_combined/egs/storage ${dir}_combined/egs/storage
  fi

  steps/nnet2/get_egs2.sh \
    --io-opts "--max-jobs-run 10" --cmd "$train_cmd" \
    --transform-dir $transform_dir \
    --left-context 14 --right-context 10 \
    $data_dir $alidir ${dir}_combined/egs
fi

if [ $stage -le 7 ]; then
  steps/nnet2/train_more.sh --learning-rate-factor 0.1 --cmd "$train_cmd" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
     ${dir}_combined_init/final.mdl ${dir}_combined/egs ${dir}_combined 
fi

if $do_decode; then
  if [ $stage -le 3 ]; then
    # do online decoding with the combined model.
    steps/nnet2/decode.sh --config conf/decode.config --skip-scoring $skip_scoring \
      --cmd "$decode_cmd" --nj $decode_nj --transform-dir "$test_transform_dir" \
      $graph_dir $test_dir ${dir}_combined/decode_`basename $test_dir`
  fi
fi

exit 0

