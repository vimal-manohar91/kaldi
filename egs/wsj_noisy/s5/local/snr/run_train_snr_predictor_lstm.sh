#!/bin/bash

# this is the lstm system, built in nnet3; 

set -ueo pipefail
. cmd.sh

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call lstm/train.sh with --gpu false

stage=0
train_stage=-10
get_egs_stage=-10
egs_opts=

# LSTM options
splice_indexes="-2,-1,0,1,2 0"
label_delay=0
num_lstm_layers=2
cell_dim=512
hidden_dim=512
recurrent_projection_dim=256
non_recurrent_projection_dim=256
chunk_width=20
chunk_left_context=40
lstm_delay="-1 -2"

# training options
num_epochs=2
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=3
num_jobs_final=8
momentum=0.5
num_chunk_per_minibatch=100
samples_per_iter=20000
remove_egs=false

# target options
train_data_dir=data/train_azteec_unsad_music_whole_sp_corrupted_hires
targets_scp=data/train_azteec_unsad_music_whole_sp_corrupted_hires/irm_targets.scp
target_type=IrmExp

config_dir=
egs_dir=
egs_suffix=

dir=exp/nnet3_irm_predictor/lstm
affix=
deriv_weights_scp=

# End configuration section.
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

num_hidden_layers=`echo $splice_indexes | perl -ane 'print scalar @F'` || exit 1

dir=$dir${affix:+_$affix}_n${num_hidden_layers}
if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi

objective_type=quadratic
if [ $target_type == "IrmExp" ]; then
  objective_type=xent
fi

num_targets=`feat-to-dim scp:$targets_scp - 2>/dev/null`

if [ $stage -le 3 ]; then
  config_extra_opts=()
  [ ! -z "$lstm_delay" ] && config_extra_opts+=(--lstm-delay="$lstm_delay")
  steps/nnet3/lstm/make_raw_configs.py "${config_extra_opts[@]}" \
    --feat-dir=$train_data_dir --num-targets=$num_targets \
    --splice-indexes="$splice_indexes" \
    --num-lstm-layers=$num_lstm_layers \
    --label-delay=$label_delay \
    --self-repair-scale=0.00001 \
    --cell-dim=$cell_dim \
    --hidden-dim=$hidden_dim \
    --recurrent-projection-dim=$recurrent_projection_dim \
    --non-recurrent-projection-dim=$non_recurrent_projection_dim \
    --include-log-softmax=false --add-lda=false \
    --add-final-sigmoid=true \
    --objective-type=$objective_type \
    --add-idct=true \
    $dir/configs
fi

if [ $stage -le 4 ]; then

  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b{05,06,11,12}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  if [ ! -z "$deriv_weights_scp" ]; then
    egs_opts="$egs_opts --deriv-weights-scp=$deriv_weights_scp"
  fi

  steps/nnet3/train_raw_rnn.py --stage=$train_stage \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.dir="$egs_dir" --egs.stage=$get_egs_stage --egs.opts="$egs_opts" \
    --trainer.num-epochs=$num_epochs \
    --trainer.samples-per-iter=$samples_per_iter \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.rnn.num-chunk-per-minibatch=$num_chunk_per_minibatch \
    --trainer.optimization.momentum=$momentum \
    --cmd="$decode_cmd" \
    --cleanup=true \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=true \
    --use-dense-targets=true \
    --feat-dir=$train_data_dir \
    --targets-scp="$targets_scp" \
    --dir=$dir  || exit 1;
fi

