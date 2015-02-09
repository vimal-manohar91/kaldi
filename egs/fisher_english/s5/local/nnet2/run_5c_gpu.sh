#!/bin/bash


# this (local/nnet2/run_5c_gpu.sh) trains a p-norm neural network on top of
# the SAT system in 4a.
# It uses the online preconditioning, which is more efficient than the
# old preconditioning.
# this script uses 8 GPUs.  
# there is no non-GPU version as it would take way too long.

dir=nnet5c_gpu
train_stage=-10
pnorm_input_dim=3000
pnorm_output_dim=300
num_hidden_layers=4
egs_dir=

. ./cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF


. utils/parse_options.sh
parallel_opts="-l gpu=1"  # This is suitable for the CLSP network, you'll likely have to change it.

dir=${dir}_i${pnorm_input_dim}_o${pnorm_output_dim}_n${num_hidden_layers}

( 
  if [ ! -f exp/$dir/final.mdl ]; then
    if [ -z "$egs_dir" ] && [ `hostname -f` == *.clsp.jhu.edu ]; then
      # spread the egs over various machines. 
      utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english_s5/exp/nnet5c_gpu/egs exp/$dir/egs/storage
    fi

    steps/nnet2/train_pnorm_simple2.sh --stage $train_stage --num-epochs 10 \
      --samples-per-iter 400000 \
      --io-opts "--max-jobs-nnet 10" \
      --num-jobs-nnet 8 --num-threads 1 \
      --minibatch-size 512 --parallel-opts "$parallel_opts" \
      --mix-up 8000 \
      --initial-learning-rate 0.08 --final-learning-rate 0.008 \
      --num-hidden-layers $num_hidden_layers \
      --pnorm-input-dim $pnorm_input_dim \
      --pnorm-output-dim $pnorm_output_dim \
      --cmd "$decode_cmd" --egs-dir "$egs_dir" \
      data/train_100k data/lang exp/tri4a exp/$dir || exit 1;
  fi

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 25 \
    --config conf/decode.config --transform-dir exp/tri4a/decode_100k_dev \
    exp/tri4a/graph_100k data/dev exp/$dir/decode_100k_dev &

)


