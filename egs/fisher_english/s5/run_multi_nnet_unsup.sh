#!/bin/bash


# this script, to be run after run_gale.sh and run_fisher.sh,
# combines all three languages.

src_dir=exp/nnet5c_gpu
ali_dir=exp/tri4a_ali_100k
egs1_dir=
egs2_dir=
initial_learning_rate=0.08
final_learning_rate=0.008
train_stage=-10
stage=-1

set -e

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

best_path_dir=${src_dir}/best_path_100k_unsup_100k_250k
dir=${src_dir}_unsup_multi_nnet_lr${initial_learning_rate}_${final_learning_rate}

if [ $stage -le -1 ]; then 
  local/best_path_weights.sh --create-ali-dir true --cmd "$decode_cmd" \
    data/unsup_100k_250k data/lang_100k_test \
    ${src_dir}/decode_100k_unsup_100k_250k \
    $best_path_dir
fi

if [ $stage -le 0 ]; then

  if [ -z "$egs1_dir" ]; then
    egs1_dir=$dir/egs1
    if [ `hostname -f` == *.clsp.jhu.edu ]; then
      # spread the egs over various machines. 
      utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english_s5/$dir/egs1 $dir/egs1/storage
    fi
  fi
  
  steps/nnet2/get_egs2.sh $egs_opts "${extra_opts[@]}" \
    --cmd "$train_cmd" --io-opts "-tc 10" \
    data/train_100k \
    $ali_dir $egs1_dir
fi

if [ $stage -le 2 ]; then
  if [ -z "$egs2_dir" ]; then
    egs2_dir=$dir/egs2
    if [ `hostname -f` == *.clsp.jhu.edu ]; then
      # spread the egs over various machines. 
      utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english_s5/$dir/egs2 $dir/egs2/storage
    fi
  fi

  extra_opts=()
  transform_dir=exp/tri4a/decode_100k_unsup_100k_250k
  extra_opts+=(--transform-dir $transform_dir)

  steps/nnet2/get_egs2.sh $egs_opts "${extra_opts[@]}" \
    --cmd "$train_cmd" --io-opts "-tc 10" \
    data/unsup_100k_250k \
    $best_path_dir $egs2_dir
fi

if [ $stage -le 3 ]; then
  if [ -z "$egs1_dir" ]; then
    egs1_dir=$dir/egs1
  fi

  if [ -z "$egs2_dir" ]; then
    egs2_dir=$dir/egs2
  fi

  steps/nnet2/train_multilang2.sh \
    --stage $train_stage --cleanup false \
    --num-epochs 20 --minibatch-size 512 \
    --initial-learning-rate $initial_learning_rate --final-learning-rate $final_learning_rate \
    --mix-up "0 12000" \
    --cmd "$train_cmd" --num-threads 1 \
    --num-jobs-nnet "4 4" --parallel-opts "-l gpu=1 -q g.q" \
    $ali_dir $egs1_dir \
    $best_path_dir $egs2_dir \
    $src_dir/100.mdl $dir
fi

if [ $stage -le 4  ]; then
  steps/nnet2/decode.sh --cmd "queue.pl" --num-threads 6 --mem $decode_mem --nj 25 \
    --config conf/decode.config --transform-dir exp/tri4a/decode_100k_dev \
    exp/tri4a/graph_100k data/dev $dir/0/decode_100k_dev &
fi

