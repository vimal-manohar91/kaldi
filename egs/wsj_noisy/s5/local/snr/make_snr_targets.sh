#!/bin/bash 

# Copyright 2015  Vimal Manohar
# Apache 2.0
set -e 
set -o pipefail

nj=4
cmd=run.pl
compress=true
data_id=
target_type=Irm
ali_rspecifier=
silence_phones_str=0
apply_exp=false
ignore_noise_dir=false
ceiling=inf
floor=-inf
stage=0
length_tolerance=2

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: $0 [options] --target-type (Irm|Snr) <clean-fbank-dir> <noise-fbank-dir> <data-dir> <log-dir> <path-to-targets>";
  echo " or  : $0 [options] --target-type FbankMask <clean-fbank-dir> <noisy-fbank-dir> <data-dir> <log-dir> <path-to-targets>";
  echo "e.g.: $0 data/train_clean_fbank data/train_noise_fbank data/train_corrupted_hires exp/make_snr_targets/train snr_targets"
  echo "options: "
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

clean_fbank_dir=$1
noise_or_noisy_fbank_dir=$2
dir=$3
tmpdir=$4
targets_dir=$5

mkdir -p $targets_dir 

[ -z "$data_id" ] && data_id=`basename $dir`

utils/split_data.sh $clean_fbank_dir $nj
utils/split_data.sh $noise_or_noisy_fbank_dir $nj

$ignore_noise_dir && utils/split_data.sh $dir $nj

targets_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $targets_dir ${PWD}`

for n in `seq $nj`; do 
  utils/create_data_link.pl $targets_dir/${data_id}.$n.ark
done

#if [ ! -z "$ali_rspecifier" ]; then
#  if ! $apply_exp; then
#    echo "If --ali-rspecifier is specified, then apply-exp must be true."
#    exit 1
#  fi
#fi

apply_exp_opts=
if $apply_exp; then
  apply_exp_opts=" copy-matrix --apply-exp=true ark:- ark:- |"
fi

if [ $stage -le 1 ]; then
  if ! $ignore_noise_dir; then
    $cmd JOB=1:$nj $tmpdir/make_`basename $targets_dir`_${data_id}.JOB.log \
      compute-snr-targets --length-tolerance=$length_tolerance --target-type=$target_type ${ali_rspecifier:+--ali-rspecifier="$ali_rspecifier" --silence-phones=$silence_phones_str} \
      --floor=$floor --ceiling=$ceiling \
      scp:$clean_fbank_dir/split$nj/JOB/feats.scp \
      scp:$noise_or_noisy_fbank_dir/split$nj/JOB/feats.scp \
      ark:- \|$apply_exp_opts \
      copy-feats --compress=$compress ark:- \
      ark,scp:$targets_dir/${data_id}.JOB.ark,$targets_dir/${data_id}.JOB.scp || exit 1
  else
    feat_dim=$(feat-to-dim scp:$clean_fbank_dir/feats.scp -) || exit 1
    $cmd JOB=1:$nj $tmpdir/make_`basename $targets_dir`_${data_id}.JOB.log \
      compute-snr-targets --length-tolerance=$length_tolerance --target-type=$target_type ${ali_rspecifier:+--ali-rspecifier="$ali_rspecifier" --silence-phones=$silence_phones_str} \
      --floor=$floor --ceiling=$ceiling --binary-targets --target-dim=$feat_dim \
      scp:$dir/split$nj/JOB/feats.scp \
      ark:- \|$apply_exp_opts \
      copy-feats --compress=$compress ark:- \
      ark,scp:$targets_dir/${data_id}.JOB.ark,$targets_dir/${data_id}.JOB.scp || exit 1
  fi
fi

for n in `seq $nj`; do
  cat $targets_dir/${data_id}.$n.scp
done > $dir/`basename $targets_dir`.scp
