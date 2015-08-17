#!/bin/bash

. cmd.sh


stage=1
train_stage=-10
use_gpu=true
initial_effective_lrate=0.005
final_effective_lrate=0.0005
pnorm_input_dim=3000
pnorm_output_dim=300
num_hidden_layers=4
egs_dir=
dir=
num_epochs=8
#splice_config="layer0/-2:-1:0:1:2 layer1/-1:2 layer3/-3:3 layer4/-7:2"
#splice_config="layer0/-2:-1:0:1:2 layer1/-1:2 layer3/-3:3"
#splice_config="layer0/-2:-1:0:1:2 layer1/-1:2 layer2/-3:3"
splice_config="layer0/-2:-1:0:1:2 layer1/-3:1 layer2/-4:2 layer3/-5:3"

set -e
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


# assume use_gpu=true since it would be way too slow otherwise.

if ! cuda-compiled; then
  cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi
parallel_opts="--gpu 1" 
num_threads=1
minibatch_size=512
[ -z "$dir" ] && dir=exp/nnet2_online/nnet_ms_a_100k_i${pnorm_input_dim}_o${pnorm_output_dim}_n${num_hidden_layers}_lr${initial_effective_lrate}_${final_effective_lrate}
mkdir -p $dir


# Stages 1 through 5 are done in run_nnet2_common.sh,
# so it can be shared with other similar scripts.
local/online/run_nnet2_common_100k.sh --stage $stage

if [ $stage -le 6 ]; then
  if [ -z "$egs_dir" ]; then # this shows how you can split across multiple file-systems.
    utils/create_split_dir.pl /export/b0{3,4,5,6}/$USER/kaldi-data/egs/fisher_english/s5/$dir/egs $dir/egs/storage
  fi

  # Because we have a lot of data here and we don't want the training to take
  # too long, we reduce the number of epochs from the defaults (15 + 5) to (3 +
  # 1).  The option "--io-opts '-tc 12'" is to have more than the default number
  # (5) of jobs dumping the egs to disk; this is OK since we're splitting our
  # data across four filesystems for speed.

  steps/nnet2/train_multisplice_accel2.sh --stage $train_stage \
    --num-epochs $num_epochs \
    --splice-indexes "$splice_config" \
    --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors_train \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "--max-jobs-run 12" \
    --num-jobs-initial 2 --num-jobs-final 14 \
    --num-hidden-layers ${num_hidden_layers} \
    --initial-effective-lrate ${initial_effective_lrate} --final-effective-lrate ${final_effective_lrate} \
    --cmd "$decode_cmd" --egs-dir "$egs_dir" \
    --pnorm-input-dim ${pnorm_input_dim} \
    --pnorm-output-dim ${pnorm_output_dim} --cleanup false \
    data/train_hires_100k data/lang exp/tri4a $dir  || exit 1;
fi

if [ $stage -le 7 ]; then
  steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
    data/lang exp/nnet2_online/extractor "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 8 ]; then
  # do the actual online decoding with iVectors, carrying info forward from 
  # previous utterances of the same speaker.
   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
      exp/tri4a/graph_100k data/dev ${dir}_online/decode_100k_dev || exit 1;
fi

#if [ $stage -le 9 ]; then
#  # this version of the decoding treats each utterance separately
#  # without carrying forward speaker information.
#   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
#     --per-utt true \
#      exp/tri4a/graph_100k data/dev ${dir}_online/decode_100k_dev_utt || exit 1;
#fi
#
#if [ $stage -le 10 ]; then
#  # this version of the decoding treats each utterance separately
#  # without carrying forward speaker information, but looks to the end
#  # of the utterance while computing the iVector.
#   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
#     --per-utt true --online false \
#      exp/tri4a/graph_100k data/dev ${dir}_online/decode_100k_dev_utt_offline || exit 1;
#fi

exit 0;
