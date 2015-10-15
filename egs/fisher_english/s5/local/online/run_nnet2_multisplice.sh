#!/bin/bash

. cmd.sh


stage=1
train_stage=-10
use_gpu=true
initial_effective_lrate=0.0015
final_effective_lrate=0.00015
presoftmax_prior_scale_power=0.0
pnorm_input_dim=3500
pnorm_output_dim=350
num_hidden_layers=6
dir=
num_epochs=10
train_data_dir=data/train_hires
lang=data/lang
graph_dir=exp/tri5a/graph
ali_dir=exp/tri4a
egs_dir=
mix_up=0

#splice_config="layer0/-2:-1:0:1:2 layer1/-1:2 layer3/-3:3 layer4/-7:2"
#splice_config="layer0/-2:-1:0:1:2 layer1/-1:2 layer3/-3:3"
#splice_config="layer0/-2:-1:0:1:2 layer1/-1:2 layer2/-3:3"
#splice_config="layer0/-2:-1:0:1:2 layer1/-1:2 layer3/-3:3 layer4/-7:2" \
splice_config="layer0/-2:-1:0:1:2 layer1/-4:-1:2 layer3/-3:3 layer4/-7:2"

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
train_id=`basename $train_data_dir`
[ -z "$dir" ] && dir=exp/nnet2_online/nnet_ms_a_${train_id}_i${pnorm_input_dim}_o${pnorm_output_dim}_n${num_hidden_layers}_lr${initial_effective_lrate}_${final_effective_lrate}_mixup${mix_up}_presoft${presoftmax_prior_scale_power}
mkdir -p $dir


# Stages 1 through 5 are done in run_nnet2_common.sh,
# so it can be shared with other similar scripts.
local/online/run_nnet2_common.sh --stage $stage

if [ $stage -le 6 ]; then
  if [ -z "$egs_dir" ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
      utils/create_split_dir.pl /export/b0{6,7,8,9}/$(USER)/kaldi-data/egs/fisher_english/s5/$dir/egs/storage $dir/egs/storage
    fi
  fi

  # Because we have a lot of data here and we don't want the training to take
  # too long, we reduce the number of epochs from the defaults (15 + 5) to (3 +
  # 1).  The option "--io-opts '-tc 12'" is to have more than the default number
  # (5) of jobs dumping the egs to disk; this is OK since we're splitting our
  # data across four filesystems for speed.


  bash -x steps/nnet2/train_multisplice_accel2.sh --stage $train_stage \
    --num-epochs $num_epochs \
    --splice-indexes "$splice_config" \
    --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors_train \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "--max-jobs-run 12" \
    --num-jobs-initial 3 --num-jobs-final 18 \
    --num-hidden-layers ${num_hidden_layers} \
    --mix-up $mix_up --presoftmax-prior-scale-power $presoftmax_prior_scale_power \
    --initial-effective-lrate ${initial_effective_lrate} --final-effective-lrate ${final_effective_lrate} \
    --cmd "$decode_cmd" --egs-dir "$egs_dir" \
    --pnorm-input-dim ${pnorm_input_dim} \
    --pnorm-output-dim ${pnorm_output_dim} --cleanup false \
    $train_data_dir $lang $ali_dir $dir  || exit 1;
fi

if [ $stage -le 7 ]; then
  steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet2_online/extractor "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 8 ]; then
  # do the actual online decoding with iVectors, carrying info forward from
  # previous utterances of the same speaker.
   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
      $graph_dir data/dev ${dir}_online/decode_`basename $lang`_dev || exit 1;
fi

exit 0

if [ $stage -le 9 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
     --per-utt true \
      $graph_dir data/dev ${dir}_online/decode_dev_utt || exit 1;
fi

if [ $stage -le 10 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information, but looks to the end
  # of the utterance while computing the iVector.
   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
     --per-utt true --online false \
      $graph_dir data/dev ${dir}_online/decode_dev_utt_offline || exit 1;
fi

exit 0;

