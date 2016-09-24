#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
train_stage=-10
get_egs_stage=-10
num_epochs=2
num_utts_subset=40     # number of utterances in validation and training
                       # subsets used for shrinkage and diagnostics.
egs_opts=
nj=20

splice_indexes="-4,-3,-2,-1,0,1,2,3,4  0  -3,1  0  -7,2 0"
relu_dims="1024 512 512 256 256 256"
relu_dim=

initial_effective_lrate=0.005
final_effective_lrate=0.0005

max_param_change=1
train_data_dir=data/train_azteec_unsad_music_whole_sp_multi_lessreverb_hires
targets_scp=data/train_azteec_unsad_music_whole_sp_multi_lessreverb_hires/irm_targets.scp
target_type=IrmExp
config_dir=
egs_dir=

dir=exp/nnet3_irm_predictor/nnet_tdnn
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

dir=${dir}_rn${num_hidden_layers}
dir=${dir}${affix}

objective_type=quadratic
if [ $target_type == "IrmExp" ]; then
  objective_type=xent
fi

num_targets=`feat-to-dim scp:$targets_scp - 2>/dev/null`

if [ -z $num_targets ]; then
  echo "Could not read num-targets" && exit 1
fi

if [ $stage -le 3 ]; then
  steps/nnet3/make_cnn_snr_predictor_configs.py \
    --feat-dir=$train_data_dir \
    --num-targets=$num_targets \
    --splice-indexes="$splice_indexes" \
    ${relu_dim:+--relu-dim=$relu_dim} \
    ${relu_dims:+--relu-dims=$relu_dims} \
    --use-presoftmax-prior-scale=false \
    --include-log-softmax=false --add-lda=false \
    --add-final-sigmoid=true \
    --objective-type=$objective_type \
    $dir/configs || exit 1
fi

if [ $stage -le 4 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b{05,06,11,12}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  if [ ! -z "$deriv_weights_scp" ]; then
    egs_opts="$egs_opts --deriv-weights-scp=$deriv_weights_scp"
  fi
  
  egs_opts="$egs_opts --num-utts-subset $num_utts_subset"

  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --egs.frames-per-eg=8 \
    --egs.dir="$egs_dir" --egs.stage=$get_egs_stage --egs.opts="$egs_opts" \
    --trainer.num-epochs=$num_epochs \
    --trainer.samples-per-iter=$samples_per_iter \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.optimization.max-param-change=$max_param_change \
    --nj=$nj --cmd="$decode_cmd" \
    --cleanup=true \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=true \
    --use-dense-targets=true \
    --feat-dir=$train_data_dir \
    --targets-scp="$targets_scp" \
    --dir=$dir  || exit 1;
fi

