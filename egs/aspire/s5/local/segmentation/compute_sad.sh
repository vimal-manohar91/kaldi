#!/bin/bash

# Copyright 2015-16  Vimal Manohar
# Apache 2.0 
set -o pipefail
set -e
set -u

. path.sh 

nj=4
cmd=run.pl
stage=-10
iter=final
extra_left_context=0
use_gpu=false
sil_prior=0.5
speech_prior=0.5
output_name=output-speech

. utils/parse_options.sh 

if [ $# -ne 3 ]; then
  echo "Usage: $0 <data-dir> <sad-model-dir> <dir>"
  echo " e.g.: $0 data/dev exp/nnet3_sad_snr/tdnn_a_n4 exp/nnet3_sad_snr/tdnn_a_n4/sad_dev"
  exit 1
fi

data=$1
model_dir=$2
dir=$3

mkdir -p $dir

echo $nj > $dir/num_jobs

gpu_cmd=$cmd
gpu_opt="--use-gpu=no"
if $use_gpu; then
  gpu_opt="--use-gpu=yes"
  gpu_cmd="$cmd --gpu 1"
fi

utils/split_data.sh $data $nj

if [ $stage -le 1 ]; then
  model=$model_dir/$iter.raw

  # Get log-posteriors from the neural network output
  $gpu_cmd JOB=1:$nj $dir/log/eval_tdnn.JOB.log \
    nnet3-compute \
    --apply-exp=false $gpu_opt \
    --extra-left-context=$extra_left_context \
    "nnet3-copy --edits=\"rename-node old-name=$output_name new-name=output\" $model - |" \
    scp:$data/split$nj/JOB/feats.scp \
    ark:$dir/log_nnet_posteriors.JOB.ark || exit 1
fi

if [ $stage -le 2 ]; then 
  post_vec=$model_dir/post.$iter.vec
  if [ ! -f $model_dir/post.$iter.vec ]; then
    if [ ! -f $model_dir/post.vec ]; then
      echo "Could not find $model_dir/post.$iter.vec. Usually computed by averaging the nnet posteriors"
      exit 1
    else
      post_vec=$model_dir/post.vec
    fi
  fi

  cat $post_vec | awk '{if (NF != 4) { print "posterior vector must have dimension two; but has dimension "NF-2; exit 1;} else { printf ("[ %f %f ]\n", log($2/($2+$3)),  log($3/($2+$3)));}}' > $dir/nnet_log_priors

  # Subtract priors to get pseudo log-likelihoods
  $cmd JOB=1:$nj $dir/log/get_likes.JOB.log \
    matrix-add-offset ark:$dir/log_nnet_posteriors.JOB.ark "vector-scale --scale=-1.0 --binary=false $dir/nnet_log_priors - |" \
    ark,scp:$dir/log_likes.JOB.ark,$dir/log_likes.JOB.scp || exit 1

  # Offset to get posteriors wrt to new priors
  cat $dir/nnet_log_priors | awk -v sil_prior=$sil_prior -v speech_prior=$speech_prior '{sum_prior = speech_prior + sil_prior; printf ("[ %f %f ]", -$2+log(sil_prior)-log(sum_prior), -$3+log(speech_prior)-log(sum_prior));}' > $dir/log_priors

  # Get new log-posteriors
  $cmd JOB=1:$nj $dir/log/adjust_priors.JOB.log \
    matrix-add-offset ark:$dir/log_nnet_posteriors.JOB.ark $dir/log_priors \
    ark,scp:$dir/log_posteriors.JOB.ark,$dir/log_posteriors.JOB.scp || exit 1

  $cmd JOB=1:$nj $dir/log/extract_logits.JOB.log \
    vector-sum "ark:extract-column --column-index=1 scp:$dir/log_posteriors.JOB.scp ark:- |" \
    "ark:extract-column --column-index=0 scp:$dir/log_posteriors.JOB.scp ark:- | vector-scale --scale=-1 ark:- ark:- |" \
    ark,scp:$dir/logits.JOB.ark,$dir/logits.JOB.scp || exit 1
fi

if [ $stage -le 3 ]; then
  $cmd JOB=1:$nj $dir/log/extract_prob.JOB.log \
    copy-matrix --apply-softmax-per-row scp:$dir/log_posteriors.JOB.scp ark:- \| \
    extract-column --column-index=1 ark:- \
    ark,scp:$dir/speech_prob.JOB.ark,$dir/speech_prob.JOB.scp || exit 1
fi

for n in `seq $nj`; do
  cat $dir/speech_prob.$n.scp
done > $dir/speech_prob.scp
