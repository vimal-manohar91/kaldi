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
num_utts_subset=300
splice_indexes="-2,-1,0,1,2"
lstm_delay=-1
num_lstm_layers=1
cell_dim=100
hidden_dim=100
recurrent_projection_dim=32
non_recurrent_projection_dim=32
chunk_width=20
chunk_left_context=100

# training options
num_epochs=8
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=3
num_jobs_final=8
label_delay=0
momentum=0.5
num_chunk_per_minibatch=100
samples_per_iter=20000

train_data_dir=data/train_si284_corrupted_hires
snr_scp=
vad_scp=
final_vad_scp=
datadir=
egs_dir=
nj=40
method=Dnn
max_param_change=1
feat_type=
config_dir=
deriv_weights_scp=
lda_opts=
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

feats_opts=(--feat-type $feat_type)
if [ "$feat_type" == "sparse" ]; then
  exit 1
fi

if [ $stage -le 3 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
      /export/b0{5,6,7,8}/$USER/kaldi-data/egs/wsj_noisy-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  deriv_weights_opt=
  if [ ! -z "$deriv_weights_scp" ]; then
    deriv_weights_opt="--deriv-weights-scp $deriv_weights_scp"
  fi

  steps/nnet3/lstm/train_raw.sh --stage $train_stage \
    --num-epochs $num_epochs --num-jobs-initial 2 --num-jobs-final 4 \
    --num-chunk-per-minibatch $num_chunk_per_minibatch \
    --samples-per-iter $samples_per_iter \
    --splice-indexes "$splice_indexes" \
    --egs-dir "$egs_dir" "${feats_opts[@]}" --num-utts-subset $num_utts_subset \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --momentum $momentum --compute-objf-opts "$compute_objf_opts" \
    --cmd "$decode_cmd" --nj 40 --objective-type linear --cleanup true \
    --max-param-change $max_param_change $deriv_weights_opt --lda-opts "$lda_opts" \
    --include-log-softmax true --skip-lda true --posterior-targets true \
    --num-lstm-layers $num_lstm_layers --lstm-delay "$lstm_delay" \
    --label-delay $label_delay \
    --cell-dim $cell_dim \
    --hidden-dim $hidden_dim \
    --recurrent-projection-dim $recurrent_projection_dim \
    --non-recurrent-projection-dim $non_recurrent_projection_dim \
    --chunk-width $chunk_width \
    --chunk-left-context $chunk_left_context \
    --egs-dir "$egs_dir" \
    --remove-egs false \
    --num-targets 2 --max-param-change $max_param_change --config-dir "$config_dir" \
    $datadir "$final_vad_scp" $dir || exit 1;
fi


