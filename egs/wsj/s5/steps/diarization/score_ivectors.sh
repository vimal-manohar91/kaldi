#!/bin/bash

# Copyright    2016  David Snyder
#              2017  Vimal Manohar
# Apache 2.0.

# TODO This script computes cosine scores from pairs of ivectors extracted
# from segments of a recording.

# Begin configuration section.
cmd="run.pl"
stage=0
target_energy=0.1
nj=10
cleanup=true
per_spk=false
pldadir=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

set -e -o pipefail -u

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0 <ivector-dir> <output-dir>"
  echo " e.g.: $0 exp/ivectors_callhome_test exp/ivectors_callhome_test"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --target-energy <target-energy|0.1>              # Target energy remaining in iVectors after applying"
  echo "                                                   # a conversation dependent PCA."
  echo "  --cleanup <bool|false>                           # If true, remove temporary files"
  exit 1;
fi

ivecdir=$1
dir=$2

mkdir -p $dir/tmp

extra_files=
if [ ! -z "$pldadir" ]; then
  extra_files="$pldadir/plda $pldadir/mean.vec $pldadir/transform.mat"
fi

for f in $ivecdir/spk2utt $ivecdir/utt2spk $ivecdir/segments $extra_files; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done
cp $ivecdir/spk2utt $dir/tmp/
cp $ivecdir/utt2spk $dir/tmp/
cp $ivecdir/segments $dir/tmp/
cp $ivecdir/spk2utt $dir/
cp $ivecdir/utt2spk $dir/
cp $ivecdir/segments $dir/

utils/data/get_reco2utt.sh $dir/tmp/

utils/spk2utt_to_utt2spk.pl $dir/tmp/reco2utt > $dir/tmp/utt2reco
utils/apply_map.pl -f 1 $dir/tmp/utt2spk < $dir/tmp/utt2reco | sort -u > $dir/tmp/spk2reco
utils/utt2spk_to_spk2utt.pl $dir/tmp/spk2reco > $dir/tmp/reco2spk

sdata=$dir/tmp/split${nj}reco;
utils/split_data.sh --per-reco $dir/tmp $nj || exit 1;

if $per_spk; then
  reco2utt="ark,t:utils/filter_scp.pl $sdata/JOB/reco2utt $dir/tmp/reco2spk |"
  if [ ! -f $ivecdir/ivector_key2samples ]; then
    ivectors="ark:utils/filter_scp.pl $sdata/JOB/spk2utt $ivecdir/ivector_spk.scp |"
  else
    for n in $(seq $nj); do 
      utils/filter_scp.pl $sdata/$n/spk2utt $ivecdir/ivector_key2samples > $sdata/$n/ivector_key2samples; 
    done
    ivectors="ark:utils/filter_scp.pl -f 2 $sdata/JOB/spk2utt $ivecdir/ivector_samples2key | utils/filter_scp.pl /dev/stdin $ivecdir/ivector_spk.scp |"
    reco2utt="$reco2utt utils/apply_map.pl -f 2- $ivecdir/ivector_key2samples |"
  fi
else
  reco2utt="ark,t:cat $sdata/JOB/reco2utt |"
  if [ ! -f $ivecdir/ivector_key2samples ]; then
    ivectors="ark:utils/filter_scp.pl $sdata/JOB/utt2spk $ivecdir/ivector.scp |"
  else
    for n in $(seq $nj); do 
      utils/filter_scp.pl $sdata/$n/utt2spk $ivecdir/ivector_key2samples > $sdata/$n/ivector_key2samples; 
    done
    ivectors="ark:utils/filter_scp.pl -f 2 $sdata/JOB/utt2spk $ivecdir/ivector_samples2key | utils/filter_scp.pl /dev/stdin $ivecdir/ivector.scp |"
    reco2utt="$reco2utt utils/apply_map.pl -f 2- $ivecdir/ivector_key2samples |"
  fi
fi

if [ ! -z "$pldadir" ]; then
  ivectors="$ivectors ivector-subtract-global-mean $pldadir/mean.vec scp:- ark:- | transform-vec $pldadir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
else
  ivectors="$ivectors copy-vector scp:- ark:- |"
fi

if [ -f $ivecdir/ivector_key2samples ]; then
  cp $ivecdir/ivector_key2samples $dir
  cp $ivecdir/ivector_samples2key $dir
else
  [ -f $dir/ivector_key2samples ] && rm $dir/ivector_key2samples
  [ -f $dir/ivector_samples2key ] && rm $dir/ivector_samples2key
fi

echo $nj > $dir/num_jobs

# Set various variables.
mkdir -p $dir/log

if [ $stage -le 0 ]; then
  echo "$0: scoring iVectors"
  $cmd JOB=1:$nj $dir/log/cosine_scoring.JOB.log \
    ivector-scoring-dense --target-energy=$target_energy \
      "$reco2utt" "$ivectors" ark,scp:$dir/scores.JOB.ark,$dir/scores.JOB.scp \
      ark,t:$dir/tmp/out_reco2utt.JOB || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining scores across jobs"
  for j in $(seq $nj); do cat $dir/scores.$j.scp; done >$dir/scores.scp || exit 1;
  for j in $(seq $nj); do cat $dir/tmp/out_reco2utt.$j; done >$dir/reco2utt || exit 1;

fi

utils/spk2utt_to_utt2spk.pl $dir/reco2utt > $dir/utt2reco
utils/filter_scp.pl $dir/utt2reco $dir/tmp/utt2spk > $dir/utt2spk
utils/filter_scp.pl $dir/utt2reco $dir/tmp/segments > $dir/segments
utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

#if $cleanup ; then
#  rm -rf $dir/tmp || exit 1;
#fi
