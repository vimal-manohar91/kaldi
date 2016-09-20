#!/bin/bash

set -o pipefail
set -e 
set -u

. cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
train_stage=-10
get_egs_stage=-10
egs_dir=
nj=20

# training options
num_epochs=6
initial_effective_lrate=0.00001
final_effective_lrate=0.0000001

train_data_dir=data/babel_mongolian_train_unsad_whole
train_data_id=babel_mongolian_train_sp_unsad_whole
vad_scp=exp/unsad_whole_data_prep_babel_mongolian_train/reco_vad/vad.scp
deriv_weights_scp=exp/unsad_whole_data_prep_babel_mongolian_train/final_vad/deriv_weights.scp

. cmd.sh
. path.sh
. ./utils/parse_options.sh

affix=_b

egs_dir=
egs_opts="--num-utts-subset 40"

dir=exp/nnet3_unsad/nnet_indomain_${train_data_id}
clean_fbank_scp=
final_vad_scp=

config_dir=     # Specify a particular config

## CNN options
## Parameter indices used for each CNN layer
## Format: layer<CNN_index>/<parameter_indices>....layer<CNN_index>/<parameter_indices>
## The <parameter_indices> for each CNN layer must contain 11 positive integers.
## The first 5 integers correspond to the parameter of ConvolutionComponent:
## <filt_x_dim, filt_y_dim, filt_x_step, filt_y_step, num_filters>
## The next 6 integers correspond to the parameter of MaxpoolingComponent:
## <pool_x_size, pool_y_size, pool_z_size, pool_x_step, pool_y_step, pool_z_step>
#cnn_indexes="6,24,2,8,256,2,5,1,1,3,1"
## Output dimension of the linear layer at the CNN output for dimension reduction
#cnn_reduced_dim=512
## Choose whether to generate delta and delta-delta features
## by adding a fixed convolution layer
#conv_add_delta=false

# DNN options
pnorm_input_dims=""
pnorm_output_dims=""
relu_dims="512 512 256 128"
splice_indexes="`seq -s , -3 3` -3,-1,1 -7,-2,2 -3,0,3"
dir=${dir}${affix}

if ! cuda-compiled; then
  cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

mkdir -p $dir

if [ -z "$final_vad_scp" ]; then 
  if [ $stage -le 1 ]; then
    mkdir -p $dir/vad/split$nj
    vad_scp_splits=()
    for n in `seq $nj`; do
      vad_scp_splits+=($dir/vad/vad.tmp.$n.scp)
    done
    utils/split_scp.pl $vad_scp ${vad_scp_splits[@]} || exit 1

    cat <<EOF > $dir/vad/vad_map
0 0
1 1
2 0
3 0
4 1
EOF
    $train_cmd JOB=1:$nj $dir/vad/log/convert_vad.JOB.log \
      copy-int-vector scp:$dir/vad/vad.tmp.JOB.scp ark,t:- \| \
      utils/apply_map.pl -f 2- $dir/vad/vad_map \| \
      copy-int-vector ark,t:- \
      ark,scp:$dir/vad/split$nj/vad.JOB.ark,$dir/vad/split$nj/vad.JOB.scp || exit 1
  fi

  for n in `seq $nj`; do
    cat $dir/vad/split$nj/vad.$n.scp
  done | sort -k1,1 > $dir/vad/vad.scp
  final_vad_scp=$dir/vad/vad.scp
fi

if [ ! -s $final_vad_scp ]; then
  echo "$0: $final_vad_scp file is empty!" && exit 1
fi

if [ $stage -le 3 ]; then
  utils/copy_data_dir.sh ${train_data_dir} ${train_data_dir}_bp_vh_hires
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires_bp.conf \
    ${train_data_dir}_bp_vh_hires exp/make_hires/${train_data_id} mfcc_hires 
  steps/compute_cmvn_stats.sh \
    ${train_data_dir}_bp_vh_hires exp/make_hires/${train_data_id} mfcc_hires 
fi

train_data_dir=${train_data_dir}_bp_vh_hires

if [ $stage -le 4 ]; then
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
    --use-mfcc "true" --minibatch-size 512 \
    --splice-indexes "$splice_indexes" --egs-dir "$egs_dir" \
    --egs-opts "$egs_opts" --frames-per-eg 8 \
    --feat-type raw --egs-dir "$egs_dir" --get-egs-stage $get_egs_stage \
    --cmvn-opts "--norm-means=false --norm-vars=false" $deriv_weights_opt \
    --max-param-change 1 \
    --initial-effective-lrate $initial_effective_lrate \
    --final-effective-lrate $final_effective_lrate \
    --cmd "$decode_cmd" --nj $nj --objective-type linear \
    --cleanup false --config-dir "$config_dir" \
    --pnorm-input-dims "$pnorm_input_dims" --pnorm-output-dims "$pnorm_output_dims" \
    --pnorm-input-dim "" --pnorm-output-dim "" \
    --relu-dims "$relu_dims" --skip-lda true \
    --posterior-targets true --include-log-softmax true --num-targets 2 \
    $train_data_dir "$final_vad_scp" $dir || exit 1;
fi

