#!/bin/bash

# Copyright     2013  Daniel Povey
#               2016  David Snyder
# Apache 2.0.

# TODO This script extracts iVectors for a set of utterances, given
# features and a trained iVector extractor.  It is based on the
# script sid/extract_ivectors.sh, but uses ivector-extract-dense
# to extract ivectors from overlapping chunks.

set -e 
set -o pipefail
set -u

# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
num_gselect=20 # Gaussian-selection using diagonal model: number of Gaussians to select
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
posterior_scale=1.0 # This scale helps to control for successve features being highly
                    # correlated.  E.g. try 0.1 or 0.3.
max_count=0
use_vad=false
pca_dim=
per_spk=false
get_multiple_samples=false
ivector_frame_rate=0.1
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: $0 <extractor-dir> <data> <ivector-dir>"
  echo " e.g.: $0 exp/extractor data/train exp/ivectors"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --num-gselect <n|20>                             # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --min-post <min-post|0.025>                      # Pruning threshold for posteriors"
  echo "  --use-vad <bool|false>                           # If true, use vad.scp instead of segments"
  exit 1;
fi

srcdir=$1
data=$2
dir=$3

for f in $srcdir/final.ie $srcdir/final.ubm $data/feats.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

if $use_vad ; then
  [ ! -f $data/vad.scp ] && echo "No such file $data/vad.scp" && exit 1;
else
  [ ! -f $data/segments ] && echo "No such file $data/segments" && exit 1;
fi
# Set various variables.

mkdir -p $dir/log
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

delta_opts=`cat $srcdir/delta_opts 2>/dev/null` || exit 1;
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null` || exit 1;
use_sliding_cmvn=`cat $srcdir/use_sliding_cmvn 2>/dev/null` || exit 1;
if [ -f $srcdir/use_perutt_cmvn ]; then
  use_perutt_cmvn=`cat $srcdir/use_perutt_cmvn 2>/dev/null` || exit 1
  f=$data/cmvn_perutt.scp
  [ ! -f $f ] && "No such file $f" && exit 1;
else
  use_perutt_cmvn=false
  f=$data/cmvn.scp
  [ ! -f $f ] && "No such file $f" && exit 1;
fi 

echo $nj > $dir/num_jobs

if $use_sliding_cmvn; then
  if $use_vad ; then
    feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding ${cmvn_opts} ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- |"
  else
    feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding ${cmvn_opts} ark:- ark:- |"
  fi
else
  utt2spk_opt=
  cmvn_scp=cmvn_perutt.scp
  if ! $use_perutt_cmvn; then
    utt2spk_opt="--utt2spk=ark,t:$sdata/JOB/utt2spk"
    cmvn_scp=cmvn.scp
  fi
  if $use_vad ; then
    feats="ark,s,cs:apply-cmvn $utt2spk_opt ${cmvn_opts} scp:$sdata/JOB/$cmvn_scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- |"
  else
    feats="ark,s,cs:apply-cmvn $utt2spk_opt ${cmvn_opts} scp:$sdata/JOB/$cmvn_scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |"
  fi
fi

ivector_suffix=
spk2utt_opt=
if $per_spk; then
  spk2utt_opt="--spk2utt=ark,t:$sdata/JOB/spk2utt"
  ivector_suffix=_spk
fi

if [ $stage -le 0 ]; then
  echo "$0: extracting iVectors"
  dubm="fgmm-global-to-gmm $srcdir/final.ubm -|"

  multiple_samples_opt=
  out_wspecifier="ark,scp:$dir/ivector${ivector_suffix}.JOB.ark,$dir/ivector${ivector_suffix}.JOB.scp"

  # If we are getting multiple samples, then write it in unpacked format.
  if $get_multiple_samples; then
    multiple_samples_opt="-multiple --ivector-frame-rate=$ivector_frame_rate"
    out_wspecifier="ark:- | unpack-matrix-into-vectors ark:- $out_wspecifier ark,t:$dir/ivector_key2samples.JOB"
  fi

  $cmd JOB=1:$nj $dir/log/extract_ivectors.JOB.log \
    gmm-gselect --n=$num_gselect "$dubm" "$feats" ark:- \| \
    fgmm-global-gselect-to-post --min-post=$min_post $srcdir/final.ubm "$feats" \
       ark,s,cs:- ark:- \| scale-post ark:- $posterior_scale ark:- \| \
    ivector-extract${multiple_samples_opt} \
      --verbose=2 --max-count=$max_count $spk2utt_opt $srcdir/final.ie \
      "$feats" ark,s,cs:- ${out_wspecifier} || exit 1
fi

if [ $stage -le 1 ]; then
  echo "$0: combining iVectors across jobs"
  for j in $(seq $nj); do cat $dir/ivector${ivector_suffix}.$j.scp; done >$dir/ivector${ivector_suffix}.scp || exit 1;
  if $get_multiple_samples; then
    for j in $(seq $nj); do cat $dir/ivector_key2samples.$j; done >$dir/ivector_key2samples
    utils/spk2utt_to_utt2spk.pl $dir/ivector_key2samples >$dir/ivector_samples2key
  fi
fi

if [ $stage -le 2 ]; then
  echo "$0: Computing mean of iVectors"
  $cmd $dir/log/mean.log \
    ivector-mean scp:$dir/ivector${ivector_suffix}.scp $dir/mean.vec || exit 1;
fi

if [ -z "$pca_dim" ]; then
  pca_dim=`cat $srcdir/ivector_dim` || exit 1
fi

if [ $stage -le 3 ]; then
  echo "$0: Computing whitening transform"
  $cmd $dir/log/transform.log \
    est-pca --read-vectors=true --normalize-mean=false \
      --normalize-variance=true --dim=$pca_dim \
      scp:$dir/ivector${ivector_suffix}.scp $dir/transform.mat || exit 1;
fi

if $per_spk; then
  if $get_multiple_samples; then 
    utils/filter_scp.pl $dir/ivector_key2samples $data/spk2utt > $dir/spk2utt
  else
    utils/filter_scp.pl $dir/ivector${ivector_suffix}.scp $data/spk2utt > $dir/spk2utt
  fi
  utils/spk2utt_to_utt2spk.pl $dir/spk2utt > $dir/utt2spk
  [ -f $dir/ivector.scp ] && rm $dir/ivector.scp
else
  if $get_multiple_samples; then 
    utils/filter_scp.pl $dir/ivector_key2samples $data/utt2spk > $dir/utt2spk
  else
    utils/filter_scp.pl $dir/ivector${ivector_suffix}.scp $data/utt2spk > $dir/utt2spk
  fi
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
  [ -f $dir/ivector_spk.scp ] && rm $dir/ivector_spk.scp
fi
utils/filter_scp.pl $dir/utt2spk $data/segments > $dir/segments
