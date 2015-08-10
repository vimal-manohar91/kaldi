#!/bin/bash

set -e 
set -o pipefail
set -u

stage=-1
train_stage=-100
use_gpu=true

srcdir=exp/multilang/tri6_nnet/0
lang=../kaldi/egs/SBS-mul-exp/data/lang/
alidir=../kaldi/egs/SBS-mul-exp/exp/tri3b
transform_dir=../kaldi/egs/SBS-mul-exp/exp/tri3b
data_dir=../kaldi/egs/SBS-mul-exp/data/train
graph_dir=../kaldi/egs/SBS-mul-exp/exp/tri3b/graph
test_dir=../kaldi/egs/SBS-mul-exp/data/eval
test_transform_dir=../kaldi/egs/SBS-mul-exp/exp/tri3b/decode_eval
mfccdir=mfcc_hires/SBS-mul-exp
tmpdir=exp/make_hires/SBS-mul-exp
dir=exp/multilang/tri6_nnet_SBS-mul-noSW

nj=20
decode_nj=5

iter=final    # iter of model to select from srcdir

# Training options
initial_learning_rate=0.005
num_epochs=25
num_epochs_combined=20

# Features options
use_hires=false
feat_type=lda

# Decoding options
acwt=0.1
scoring_script=local/score.sh
scoring_opts="--min-lmwt 1 --max-lmwt 15"
skip_scoring=false

debug_mode=false
do_decode=false

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

if $use_hires; then
  data_id=`basename $data_dir`
  test_id=`basename $test_dir`

  if [ $stage -le -2 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
      date=$(date +'%m_%d_%H_%M')
      utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/babel-$date/s5c/$mfccdir/storage $mfccdir/storage
    fi

    utils/copy_data_dir.sh $data_dir ${data_dir}_hires
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" ${data_dir}_hires $tmpdir/$data_id $mfccdir || exit 1
    steps/compute_cmvn_stats.sh ${data_dir}_hires $tmpdir/$data_id $mfccdir || exit 1
  fi
  data_dir=${data_dir}_hires
  
  if [ $stage -le -1 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
      date=$(date +'%m_%d_%H_%M')
      utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/babel-$date/s5c/$mfccdir/storage $mfccdir/storage
    fi

    utils/copy_data_dir.sh $test_dir ${test_dir}_hires
    steps/make_mfcc.sh --nj $decode_nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" ${test_dir}_hires $tmpdir/$test_id $mfccdir || exit 1
    steps/compute_cmvn_stats.sh ${test_dir}_hires $tmpdir/$test_id $mfccdir || exit 1
  fi
  test_dir=${test_dir}_hires
fi

if [ $stage -le 0 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir//egs/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/babel-$date/s5c/$dir/egs/storage ${dir}/egs/storage
  fi

  cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1;
  if [ -z "$feat_type" ]; then
    if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=raw; fi
    echo "$0: feature type is $feat_type"
  fi

  steps/nnet2/get_egs2.sh \
    --io-opts "--max-jobs-run 10" --cmd "$train_cmd" \
    --cmvn-opts "$cmvn_opts" \
    --feat-type $feat_type \
    --transform-dir "$transform_dir" \
    --left-context 14 --right-context 10 \
    $data_dir $alidir $dir/egs
fi

if [ $stage -le 1 ]; then
  learning_rates=$(nnet-am-info --print-learning-rates $srcdir/final.mdl 2>/dev/null | perl -an -F/:/ -e "for (\$i=0; \$i < \$#F; \$i++) { print \"0:\"}; print $initial_learning_rate") || exit 1

  steps/nnet2/train_more2.sh \
    --reinitialize-softmax-model $alidir/final.mdl \
    --initial-learning-rates $learning_rates \
    --learning-rate-factor 0.9 --cmd "$train_cmd" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-jobs-nnet 4 \
    --num-epochs $num_epochs \
     ${srcdir}/final.mdl $dir/egs $dir
fi

graph_id=${graph_dir#*graph}
if $do_decode; then
  if [ $stage -le 3 ]; then
    steps/nnet2/decode.sh --config conf/decode.config --skip-scoring $skip_scoring \
      --cmd "$decode_cmd" --nj $decode_nj --transform-dir "$test_transform_dir" --acwt $acwt \
      $graph_dir $test_dir ${dir}/decode${graph_id}_`basename $test_dir` || exit 1
  fi

  if [ $stage -le 4 ]; then
    $scoring_script $scoring_opts --cmd "$decode_cmd" \
      $test_dir $graph_dir $dir/decode${graph_id}_`basename $test_dir` || exit 1
  fi
fi

if $debug_mode; then
  echo "Exiting because --debug-mode true" && exit 0
fi

if [ $stage -le 7 ]; then
  learning_rates=$(nnet-am-info --print-learning-rates $srcdir/final.mdl 2>/dev/null) || exit 1
  steps/nnet2/train_more2.sh --initial-learning-rates $learning_rates \
    --learning-rate-factor 0.1 --cmd "$train_cmd" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-epochs $num_epochs_combined \
     ${dir}/final.mdl ${dir}/egs ${dir}_combined 
fi

graph_id=${graph_dir#*graph}
if $do_decode; then
  if [ $stage -le 8 ]; then
    steps/nnet2/decode.sh --config conf/decode.config --skip-scoring $skip_scoring \
      --cmd "$decode_cmd" --nj $decode_nj --transform-dir "$test_transform_dir" --acwt $acwt \
      $graph_dir $test_dir ${dir}_combined/decode${graph_id}_`basename $test_dir` || exit 1
  fi

  if [ $stage -le 9 ]; then
    $scoring_script $scoring_opts \
      $test_dir $graph_dir ${dir}_combined/decode${graph_id}_`basename $test_dir` || exit 1
  fi
fi

exit 0
