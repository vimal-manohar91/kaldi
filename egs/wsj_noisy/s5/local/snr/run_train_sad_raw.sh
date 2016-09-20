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
nj=20

# training options
num_epochs=2
initial_effective_lrate=0.00001
final_effective_lrate=0.0000001

. cmd.sh
. path.sh
. ./utils/parse_options.sh

dir=exp/nnet3_unsad_music/nnet_raw
affix=_a

egs_dir=
egs_opts="--num-utts-subset 20"

train_data_dir=data/train_azteec_sp_unsad_music_whole_multi_hires/
vad_scp=exp/unsad_music_whole_data_prep_train_100k_sp/reco_vad/vad_train_azteec_multi.scp
deriv_weights_scp=exp/unsad_music_whole_data_prep_train_100k_sp/final_vad/deriv_weights_azteec_multi.scp
clean_fbank_scp=
final_vad_scp=

splice_indexes="-2,-1,0,1,2 0 0 -2,1 -1,2 -3,3 -5,2 -7,3 -9,-5,0,3"

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
  steps/nnet3/make_jesus_snr_predictor_configs.py \
    --feat-dir=$train_data_dir \
    --num-targets=2 \
    --splice-indexes="$splice_indexes" \
    --jesus.layer="--hidden-dim=1800 --forward-input-dim=500 --forward-output-dim=500 --num-blocks=100" \
    --feature-extraction.jesus-layer="--hidden-dim=1800 --forward-input-dim=500 --forward-output-dim=500 --num-blocks=100" \
    --feat-type="waveform" \
    --feature-extraction.max-shift=0.2 \
    --feature-extraction.conv-filter-dim=250 \
    --feature-extraction.conv-filter-step=10 \
    --feature-extraction.conv-num-filters=100 \
    --feature-extraction.num-hidden-layers=2 \
    --use-presoftmax-prior-scale=false \
    --add-lda=false \
    --include-log-softmax=true $dir/configs || exit 1
fi

if [ $stage -le 4 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  if [ ! -z "$deriv_weights_scp" ]; then
    egs_opts="$egs_opts --deriv-weights-scp $deriv_weights_scp"
  fi

  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --feat-cmvn-opts="--norm-means=false --norm-vars=false" \
    --egs.frames-per-eg=150 \
    --egs.dir="$egs_dir" --egs.stage=$get_egs_stage --egs-opts="$egs_opts" \
    --trainer.num-epochs=$num_epochs \
    --trainer.shuffle-buffer-size=1000 \
    --trainer.add-layers-period=3 \
    --trainer.max-param-change=2 \
    --trainer.optimization.minibatch-size=128 \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=14 \
    --cleanup=true \
    --cleanup.remove-egs=false \
    --cleanup.preserve-model-interval=10 \
    --use-dense-targets=false \
    --feat-dir=$train_data_dir \
    --targets-scp="$final_vad_scp" \
    --dir=$dir || exit 1
fi
