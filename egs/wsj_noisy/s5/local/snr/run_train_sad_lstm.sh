#!/bin/bash

# this is the lstm system, built in nnet3; 

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

# LSTM options
splice_indexes="-2,-1,0,1,2 0"
label_delay=0
num_lstm_layers=2
cell_dim=256
hidden_dim=256
recurrent_projection_dim=128
non_recurrent_projection_dim=128
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
max_param_change=1

# target options
train_data_dir=data/train_azteec_unsad_music_whole_sp_multi_lessreverb_1k_hires
snr_scp=
vad_scp=
final_vad_scp=
datadir=
egs_dir=
nj=40
feat_type=raw
config_dir=
deriv_weights_scp=
compute_objf_opts=

dir=
affix=a

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

num_hidden_layers=`echo $splice_indexes | perl -ane 'print scalar @F'` || exit 1
if [ -z "$dir" ]; then
  dir=exp/nnet3_sad_snr/nnet_lstm
fi

dir=$dir${affix:+_$affix}_n${num_hidden_layers}
if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi


if ! cuda-compiled; then
  cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi
  
mkdir -p $dir

datadir=${train_data_dir}

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
  config_extra_opts=()
  [ ! -z "$lstm_delay" ] && config_extra_opts+=(--lstm-delay="$lstm_delay")
  steps/nnet3/lstm/make_raw_configs.py "${config_extra_opts[@]}" \
    --feat-dir=$train_data_dir --num-targets=2 \
    --splice-indexes="$splice_indexes" \
    --num-lstm-layers=$num_lstm_layers \
    --label-delay=$label_delay \
    --self-repair-scale=0.00001 \
    --cell-dim=$cell_dim \
    --hidden-dim=$hidden_dim \
    --recurrent-projection-dim=$recurrent_projection_dim \
    --non-recurrent-projection-dim=$non_recurrent_projection_dim \
    --include-log-softmax=true --add-lda=false \
    --add-final-sigmoid=false \
    --objective-type=linear \
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

  egs_opts="$egs_opts --num-utts-subset $num_utts_subset"

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
    --trainer.max-param-change=$max_param_change \
    ${config_dir:+--configs-dir=$config_dir} \
    --cmd="$decode_cmd" --nj 40 \
    --cleanup=true \
    --cleanup.remove-egs=$remove_egs \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=true \
    --use-dense-targets=false \
    --feat-dir=$datadir \
    --targets-scp="$final_vad_scp" \
    --dir=$dir || exit 1
fi

