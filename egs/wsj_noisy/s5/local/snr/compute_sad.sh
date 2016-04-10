#!/bin/bash

# Copyright 2015  Vimal Manohar
# Apache 2.0 
set -o pipefail
set -e
set -u

. path.sh 

method=Dnn
nj=4
cmd=run.pl
stage=-10
iter=final
splice_opts="--left-context=10 --right-context=10"
model_dir=exp/nnet3_sad_snr/tdnn_train_si284_corrupted_splice21
snr_pred_dir=exp/frame_snrs_lwr_snr_reverb_dev_aspire_whole/
dir=exp/nnet3_sad_snr/sad_train_si284_corrupted
extra_left_context=0
use_gpu=false
sil_prior=0.5
speech_prior=0.5
add_frame_snr=false
snr_data_dir=

. utils/parse_options.sh 

if [ $# -ne 3 ]; then
  echo "Usage: $0 <sad-model-dir> <snr-pred-dir> <dir>"
  echo " e.g.: $0 $model_dir $snr_pred_dir $dir"
  exit 1
fi

model_dir=$1
snr_pred_dir=$2
dir=$3

mkdir -p $dir

#feat_type=`cat $model_dir/feat_type` || exit 1

echo $nj > $dir/num_jobs

gpu_cmd=$cmd
gpu_opt="--use-gpu=no"
if $use_gpu; then
  gpu_opt="--use-gpu=yes"
  gpu_cmd="$cmd --gpu 1"
fi

append_feats_opts="copy-feats scp:- ark:- |"


if [ -z "$snr_data_dir" ]; then
  if [ ! -s $snr_pred_dir/nnet_pred_snrs.scp ]; then  
    echo "$0: Could not read $snr_pred_dir/nnet_pred_snrs.scp or it is empty" 
    exit 1
  fi
  if $add_frame_snr; then
    append_feats_opts="append-feats scp:- scp:$snr_pred_dir/frame_snrs.scp ark:- |"
  fi
fi

feats=$snr_pred_dir/nnet_pred_snrs.scp
if [ ! -z "$snr_data_dir" ]; then
  feats=$snr_data_dir/feats.scp
fi

if [ $stage -le 1 ]; then
  case $method in 
    "LogisticRegressionSubsampled")
      model=$model_dir/$iter.mdl

      $cmd --mem 8G JOB=1:$nj $dir/log/eval_logistic_regression.JOB.log \
        logistic-regression-eval-on-feats "$model" \
        "ark:utils/split_scp.pl -j $nj \$[JOB-1] $feats |$append_feats_opts splice-feats $splice_opts ark:- ark:- |" \
        ark:$dir/log_nnet_posteriors.JOB.ark || exit 1
      ;;
    "LogisticRegression"|"Dnn")
      model=$model_dir/$iter.raw

      $gpu_cmd JOB=1:$nj $dir/log/eval_tdnn.JOB.log \
        nnet3-compute --apply-exp=false $gpu_opt \
        --extra-left-context=$extra_left_context "$model" \
        "ark:utils/split_scp.pl -j $nj \$[JOB-1] $feats |$append_feats_opts" \
        ark:$dir/log_nnet_posteriors.JOB.ark || exit 1
      ;;
    *)
      echo "Unknown method $method" 
      exit 1
  esac
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

  $cmd JOB=1:$nj $dir/log/get_likes.JOB.log \
    matrix-add-offset ark:$dir/log_nnet_posteriors.JOB.ark "vector-scale --scale=-1.0 --binary=false $dir/nnet_log_priors - |" \
    ark,scp:$dir/log_likes.JOB.ark,$dir/log_likes.JOB.scp || exit 1

  cat $dir/nnet_log_priors | awk -v sil_prior=$sil_prior -v speech_prior=$speech_prior '{sum_prior = speech_prior + sil_prior; printf ("[ %f %f ]", -$2+log(sil_prior)-log(sum_prior), -$3+log(speech_prior)-log(sum_prior));}' > $dir/log_priors

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
    loglikes-to-post scp:$dir/log_posteriors.JOB.scp ark:- \| \
    weight-pdf-post 0 0 ark:- ark:- \| post-to-weights ark:- \
    ark,scp:$dir/speech_prob.JOB.ark,$dir/speech_prob.JOB.scp || exit 1
fi
