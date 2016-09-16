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
chunk_width=20
chunk_left_context=40

# target options
train_data_dir=data/train_azteec_unsad_music_whole_sp_corrupted_hires
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

snr_predictor_dir=exp/nnet3/lstm_n2
sad_model_dir=exp/nnet3_sad_snr/nnet_lstm_a_n2

snr_predictor_iter=final
sad_model_iter=final
joint_iter=final

train_snr_predictor=true
joint_training=false

dir=
affix=a

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ -z "$dir" ]; then
  dir=exp/nnet3_sad_snr/nnet_joint
fi

dir=$dir${affix:+_$affix}

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

sad_model=$sad_model_dir/$sad_model_iter.raw
snr_predictor_model=$snr_predictor_dir/$snr_predictor_iter.raw

if [ $stage -le 3 ]; then
  snr_predictor_output=`nnet3-info --print-args=false $snr_predictor_model | perl -ne '/output-node name=output input=(\S+)/ && print $1'`

  targets_dim=`nnet3-info --print-args=false $snr_predictor_model | perl -ne '/output-node name=output input=\S+ dim=(\d+)/ && print $1'`
  clean_feat_dim=`nnet3-info --print-args=false $sad_model | perl -ne '/input-node name=input dim=(\d+)/ && print $1'`

  if [ -z "$snr_predictor_output" ]; then
    echo "snr-predictor network does not have the required structure"
    exit 1
  fi

  cp -rT $sad_model_dir/configs $dir/configs

  python utils/get_dct_matrix.py \
    --num-ceps=$clean_feat_dim --num-filters=$targets_dim \
    $dir/configs/dct.mat

  python utils/get_dct_matrix.py \
    --num-ceps=$clean_feat_dim --num-filters=$targets_dim \
    --get-idct-matrix=true \
    $dir/configs/idct.mat

  log_floor=0.0001
  output_scale=1

  cat <<EOF > $dir/configs/clean_pred.config 
component name=log_irm type=LogComponent dim=$targets_dim log-floor=$log_floor
component-node name=log_irm component=log_irm input=$snr_predictor_output 
component name=scaled_log_irm type=FixedScaleComponent dim=$targets_dim scale=$output_scale
component-node name=scaled_log_irm component=scaled_log_irm input=log_irm
component name=fbank type=FixedAffineComponent matrix=$dir/configs/idct.mat
component-node name=fbank component=fbank input=input
component name=clean_pred type=FixedAffineComponent matrix=$dir/configs/dct.mat
component-node name=clean_pred component=clean_pred input=Sum(fbank, scaled_log_irm)
output-node name=output input=clean_pred
EOF

  nnet3-copy \
    --binary=false $sad_model $dir/sad_init.raw

  nnet3-init $snr_predictor_model $dir/configs/clean_pred.config \
    $dir/snr_predictor_init.raw
fi

if [ $stage -le 4 ]; then
  snr_predictor_model=$dir/snr_predictor_init.raw
  sad_model=$dir/sad_init.raw

  snr_predictor_lrate_factors=`nnet3-info --print-args=false --print-learning-rates $snr_predictor_model | \
  python -c '
import sys
second = lambda x:x.split(":")[1] 
for line in sys.stdin.readlines():
  splits = line.strip().split()
  if splits[0] == "learning-rate-factors:": 
    print (":".join([ second(x) for x in splits[2:-1] ]))'`

  sad_model_lrate_factors=`nnet3-info --print-args=false --print-learning-rates $sad_model | \
  python -c '
import sys
second = lambda x:x.split(":")[1] 
for line in sys.stdin.readlines():
  splits = line.strip().split()
  if splits[0] == "learning-rate-factors:": 
    print (":".join([ second(x) for x in splits[2:-1] ]))'`

  if $train_snr_predictor; then
    lrate_factors=`echo $sad_model_lrate_factors | perl -a -F: -ne 'print join(":", (0)x(scalar @F))'`
    nnet3-copy --add-prefix-to-names="n0-" $snr_predictor_model $dir/snr_predictor_temp.raw
    nnet3-copy --learning-rate-factors=$lrate_factors --add-prefix-to-names="n1-" \
      $sad_model - | nnet3-copy --rename-nodes-wxfilename="echo n1-input n0-clean_pred |" \
      --binary=false - $dir/sad_model_temp.raw
  else
    lrate_factors=`echo $snr_predictor_lrate_factors | perl -a -F: -ne 'print join(":", (0)x(scalar @F))'`
    nnet3-copy --add-prefix-to-names="n0-" --learning-rate-factors=$lrate_factors $snr_predictor_model $dir/snr_predictor_temp.raw
    nnet3-copy --add-prefix-to-names="n1-" $sad_model - | \
      nnet3-copy --rename-nodes-wxfilename="echo n1-input n0-clean_pred |" \
      --binary=false - $dir/sad_model_temp.raw
  fi

  nnet3-append-nnets - $dir/snr_predictor_temp.raw \
    "awk '!/input-node name=n0-clean_pred/ {print}' $dir/sad_model_temp.raw |" | \
    nnet3-copy --rename-nodes-wxfilename="echo -e 'n1-output output\nn0-input input' |" - $dir/init.raw
fi

if [ $stage -le 5 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
      /export/b0{5,6,7,8}/$USER/kaldi-data/egs/wsj_noisy-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  if [ ! -z "$deriv_weights_scp" ]; then
    egs_opts="$egs_opts --deriv-weights-scp $deriv_weights_scp"
  fi

  egs_opts="$egs_opts --num-utts-subset $num_utts_subset"

  steps/nnet3/train_raw_more.py --stage=$train_stage \
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

[ -z "$egs_dir" ] && egs_dir=$dir/egs
initial_effective_lrate=0.00003
final_effective_lrate=0.000003
 
if $joint_training; then
  joint_model=$dir/$joint_iter.raw

  mkdir -p ${dir}_joint
  cp -r $dir/configs ${dir}_joint

  dir=${dir}_joint

  lrate_factors=`nnet3-info --print-args=false --print-learning-rates $joint_model | \
    python -c '
import sys
first = lambda x:x.split(":")[0] 
second = lambda x:x.split(":")[1] 
for line in sys.stdin.readlines():
  splits = line.strip().split()
  if splits[0] == "learning-rate-factors:": 
    print (":".join([ "1.0" for x in splits[2:-1] ]))'`

  nnet3-copy --learning-rate-factors=$lrate_factors $joint_model $dir/init.raw

  if [ $stage -le 6 ]; then
    steps/nnet3/train_raw_more.py --stage=$train_stage \
      --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
      --egs.chunk-width=$chunk_width \
      --egs.chunk-left-context=$chunk_left_context \
      --egs.dir="$egs_dir" --egs.stage=100 --egs.opts="$egs_opts" \
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
fi

