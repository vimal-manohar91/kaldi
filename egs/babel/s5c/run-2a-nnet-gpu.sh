#!/bin/bash
dir=exp/tri6_nnet
train_stage=-10
train_data=data/train
lang=data/lang
ali_dir=exp/tri5_ali
L=ASM

. conf/common_vars.sh

# This parameter will be used when the training dies at a certain point.
train_stage=-100
. ./utils/parse_options.sh

. ./langconf/$L/lang.conf

set -e
set -o pipefail
set -u

# Wait till the main run.sh gets to the stage where's it's 
# finished aligning the tri5 model.
echo "Waiting till $ali_dir/.done exists...."
while [ ! -f $ali_dir/.done ]; do sleep 30; done
echo "...done waiting for $ali_dir/.done"

if [ ! -f $dir/.done ]; then
  steps/nnet2/train_pnorm_fast.sh \
    --stage $train_stage --mix-up $dnn_mixup \
    --initial-learning-rate $dnn_init_learning_rate \
    --final-learning-rate $dnn_final_learning_rate \
    --num-hidden-layers $dnn_num_hidden_layers \
    --pnorm-input-dim $dnn_input_dim \
    --pnorm-output-dim $dnn_output_dim \
    --cmd "$train_cmd" \
    "${dnn_gpu_parallel_opts[@]}" \
    $train_data $lang $ali_dir $dir || exit 1

  touch $dir/.done
fi

