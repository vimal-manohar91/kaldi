#!/bin/bash

# this is the lstm system, built in nnet3; 

. cmd.sh

# this is a basic lstm script
# LSTM script runs for more epochs than the TDNN script
# and each epoch takes twice the time

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call lstm/train.sh with --gpu false

stage=0
train_stage=-10
get_egs_stage=-10
num_epochs=8

# LSTM options
splice_indexes="-2,-1,0,1,2 0 0"
label_delay=5
num_lstm_layers=3
cell_dim=1024
hidden_dim=1024
recurrent_projection_dim=256
non_recurrent_projection_dim=256
chunk_width=20
chunk_left_context=40

# training options
num_epochs=8
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=3
num_jobs_final=8
momentum=0.5
num_chunk_per_minibatch=100
samples_per_iter=20000
remove_egs=false

#decode options
extra_left_context=
frames_per_chunk=

# target options
train_data_dir=data/train_si284_corrupted_hires
targets_scp=data/train_si284_corrupted_hires/snr_targets.scp
target_type=IrmExp

config_dir=
egs_dir=
egs_suffix=

dir=exp/nnet3/lstm
affix=

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

if [ $stage -le 8 ]; then
  echo $target_type > $dir/target_type

  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/lstm/train_raw.sh --stage $train_stage \
    --label-delay $label_delay \
    --num-epochs $num_epochs --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
    --num-chunk-per-minibatch $num_chunk_per_minibatch \
    --samples-per-iter $samples_per_iter \
    --splice-indexes "$splice_indexes" --egs-suffix "$egs_suffix" \
    --feat-type raw --get-egs-stage $get_egs_stage \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --momentum $momentum \
    --cmd "$decode_cmd" --nj 40 --objective-type $objective_type \
    --cleanup false --config-dir "$config_dir" \
    --num-lstm-layers $num_lstm_layers \
    --cell-dim $cell_dim \
    --hidden-dim $hidden_dim \
    --recurrent-projection-dim $recurrent_projection_dim \
    --non-recurrent-projection-dim $non_recurrent_projection_dim \
    --chunk-width $chunk_width \
    --chunk-left-context $chunk_left_context \
    --egs-dir "$egs_dir" \
    --remove-egs $remove_egs \
    $train_data_dir $targets_scp $dir || exit 1
fi


