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

# training options
num_epochs=2
initial_effective_lrate=0.00001
final_effective_lrate=0.0000001

minibatch_size=256
add_layers_period=2

# Add 2 (--feature-extraction.num-hidden-layers) jesus layers after input
pooling_type=jesus  
feat_extract_jesus_config="--hidden-dim=3040 --forward-input-dim=380 --forward-output-dim=380 --num-blocks=38 --use-repeated-affine=false"

# Here, first 2 splices correspond to the jesus layers
splice_indexes="-3,-2,-1,0,1,2,3 -3,0,2 -3,0 -5,1 -7,2 -9,-5,0,3" 

# To not have any jesus layers use
# pooling_type=none
# feat_extract_jesus_config=""

# For other options, see steps/nnet3/make_jesus_snr_predictor_configs.py.

relu_dim=2000
chunk_left_context=40   # With this context, the created egs can also be used for LSTM training
chunk_right_context=5   
frames_per_eg=8
nj=20

dir=exp/nnet3_unsad_music/nnet_raw
affix=_a

egs_dir=

train_data_dir=data/train_azteec_unsad_music_whole_sp_multi_lessreverb_hires/
vad_scp=exp/unsad_music_whole_data_prep_train_100k_sp/reco_vad/vad_azteec_multi.scp
deriv_weights_scp=exp/unsad_music_whole_data_prep_train_100k_sp/final_vad/deriv_weights_azteec_multi.scp
clean_fbank_scp=
final_vad_scp=

. cmd.sh
. path.sh
. ./utils/parse_options.sh

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
    --cnn.cepstral-lifter=0 \
    --num-targets=2 \
    --splice-indexes="$splice_indexes" \
    --relu-dim=$relu_dim \
    --feature-extraction.pooling-type="$pooling_type" \
    --feature-extraction.jesus-layer="$feat_extract_jesus_config" \
    --feature-extraction.num-hidden-layers=2 \
    --feat-type="mfcc" \
    --use-presoftmax-prior-scale=false \
    --add-lda=false \
    --include-log-softmax=true $dir/configs || exit 1
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
    --egs.frames-per-eg=$frames_per_eg \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.dir="$egs_dir" --egs.stage=$get_egs_stage --egs.opts="$egs_opts" \
    --egs.cmd="$decode_cmd --mem 12G" \
    --cmd="$decode_cmd" \
    --trainer.num-epochs=$num_epochs \
    --trainer.add-layers-period=$add_layers_period \
    --trainer.max-param-change=2 \
    --trainer.optimization.minibatch-size=$minibatch_size \
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

