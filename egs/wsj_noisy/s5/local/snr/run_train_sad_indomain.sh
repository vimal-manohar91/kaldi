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
egs_opts=
num_utts_subset=40

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

affix=b

. cmd.sh
. path.sh
. ./utils/parse_options.sh

dir=exp/nnet3_unsad/nnet_indomain_${train_data_id}
clean_fbank_scp=
final_vad_scp=

config_dir=     # Specify a particular config

# DNN options
relu_dims="512 512 256 128"
splice_indexes="`seq -s , -3 3` -3,-1,1 -7,-2,2 -3,0,3"
dir=$dir${affix:+_$affix}

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
  steps/nnet3/make_cnn_snr_predictor_configs.py \
    --feat-dir=$train_data_dir --num-targets=2 \
    --splice-indexes="$splice_indexes" \
    --self-repair-scale=0.00001 \
    ${relu_dim:+--relu-dim=$relu_dim} \
    ${relu_dims:+--relu-dim=$relu_dims} \
    --include-log-softmax=true --add-lda=false \
    --add-final-sigmoid=false \
    --objective-type=linear \
    $dir/configs

fi

if [ $stage -le 5 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
      /export/b{05,06,11,12}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  if [ ! -z "$deriv_weights_scp" ]; then
    egs_opts="$egs_opts --deriv-weights-scp $deriv_weights_scp"
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
    --trainer.max-param-change=$max_param_change \
    ${config_dir:+--configs-dir=$config_dir} \
    --cmd="$decode_cmd" --nj 40 \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=true \
    --use-dense-targets=false \
    --feat-dir=$datadir \
    --targets-scp="$final_vad_scp" \
    --dir=$dir || exit 1
fi

