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
num_epochs=8
egs_opts=
nj=20

# CNN options
# Parameter indices used for each CNN layer
# Format: layer<CNN_index>/<parameter_indices>....layer<CNN_index>/<parameter_indices>
# The <parameter_indices> for each CNN layer must contain 11 positive integers.
# The first 5 integers correspond to the parameter of ConvolutionComponent:
# <filt_x_dim, filt_y_dim, filt_x_step, filt_y_step, num_filters>
# The next 6 integers correspond to the parameter of MaxpoolingComponent:
# <pool_x_size, pool_y_size, pool_z_size, pool_x_step, pool_y_step, pool_z_step>
cnn_indexes="3,8,1,1,256,1,3,1,1,3,1"
# Output dimension of the linear layer at the CNN output for dimension reduction
cnn_reduced_dim=256
# Choose whether to generate delta and delta-delta features
# by adding a fixed convolution layer
conv_add_delta=false

splice_indexes="-4,-3,-2,-1,0,1,2,3,4  0  -3,1  0  -7,2 0"
initial_effective_lrate=0.005
final_effective_lrate=0.0005
pnorm_input_dims="3000 3000 3000 3000 3000 3000"
pnorm_output_dims="300 300 300 300 300 300"
relu_dims=
train_data_dir=data/train_si284_corrupted_hires
targets_scp=data/train_si284_corrupted_hires/snr_targets.scp
max_param_change=1
add_layers_period=2
target_type=IrmExp
config_dir=
egs_dir=
egs_suffix=
src_dir=
src_iter=final
dir=
affix=
deriv_weights_scp=
clean_fbank_scp=

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

num_hidden_layers=`echo $splice_indexes | perl -ane 'print scalar @F'` || exit 1
if [ -z "$dir" ]; then
  dir=exp/nnet3_snr_predictor/nnet_tdnn_a
fi

if [ -z "$relu_dims" ]; then
dir=${dir}_pn${num_hidden_layers}_lrate${initial_effective_lrate}_${final_effective_lrate}
else
dir=${dir}_rn${num_hidden_layers}_lrate${initial_effective_lrate}_${final_effective_lrate}
fi

dir=${dir}${affix}

if ! cuda-compiled; then
  cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

objective_type=quadratic
if [ $target_type == "IrmExp" ]; then
  objective_type=xent
fi

mkdir  -p $dir

if [ $stage -le 8 ]; then
  echo $target_type > $dir/target_type

  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj_noisy-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  deriv_weights_opt=
  if [ ! -z "$deriv_weights_scp" ]; then
    deriv_weights_opt="--deriv-weights-scp $deriv_weights_scp"
  fi

  steps/nnet3/train_tdnn_raw.sh --stage $train_stage \
    --num-epochs $num_epochs --num-jobs-initial 2 --num-jobs-final 14 \
    --cnn-indexes "$cnn_indexes" \
    --cnn-reduced-dim "$cnn_reduced_dim" \
    --conv-add-delta $conv_add_delta \
    --use-mfcc "true" --minibatch-size 128 \
    --splice-indexes "$splice_indexes" \
    --egs-suffix "$egs_suffix" --egs-opts "$egs_opts" \
    --feat-type raw --egs-dir "$egs_dir" --get-egs-stage $get_egs_stage \
    --cmvn-opts "--norm-means=false --norm-vars=false" $deriv_weights_opt \
    --max-param-change $max_param_change \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --cmd "$decode_cmd" --nj $nj --objective-type $objective_type --cleanup false --config-dir "$config_dir" \
    --pnorm-input-dims "$pnorm_input_dims" --pnorm-output-dims "$pnorm_output_dims" --pnorm-input-dim "" --pnorm-output-dim "" \
    --relu-dims "$relu_dims" --l2-regularizer-targets "$clean_fbank_scp" \
    --add-layers-period $add_layers_period \
    $train_data_dir $targets_scp $dir || exit 1;
fi

