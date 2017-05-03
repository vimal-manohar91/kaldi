#!/bin/bash

# Copyright    2016  David Snyder
# Apache 2.0.

# TODO This script computes PLDA scores from pairs of ivectors extracted
# from segments of a recording.

# Begin configuration section.
cmd="run.pl"
stage=0
target_energy=0.1
nj=10
cleanup=true
use_src_mean=false
use_src_transform=false
per_spk=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

set -e -o pipefail -u

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: $0 <plda-dir> <ivector-dir> <output-dir>"
  echo " e.g.: $0 exp/ivectors_callhome_heldout exp/ivectors_callhome_test exp/ivectors_callhome_test"
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

pldadir=$1
ivecdir=$2
dir=$3

mkdir -p $dir/tmp

for f in $ivecdir/spk2utt $ivecdir/utt2spk $ivecdir/segments $pldadir/plda; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

do_sph_norm=false
do_efr_norm=false
if [ -f $pldadir/snn/transform_iter0.mat ]; then
  do_sph_norm=true
  norm_dir=$pldadir/snn
elif [ -f $pldadir/efr/transform_iter0.mat ]; then
  do_efr_norm=true
  norm_dir=$pldadir/efr
fi

for f in $pldadir/mean.vec $pldadir/transform.mat; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

if ! $do_sph_norm && ! $do_efr_norm; then
  for f in $pldadir/mean.vec $pldadir/transform.mat; do
    [ ! -f $f ] && echo "No such file $f" && exit 1;
  done

  if $use_src_mean; then
    if $use_src_transform; then
      mkdir -p $dir/plda_src_mean_tx

      cp $pldadir/plda $dir/plda_src_mean_tx
      cp $ivecdir/mean.vec $dir/plda_src_mean_tx

      cp $ivecdir/transform.mat $dir/plda_src_mean_tx

      for f in $pldadir/spk2utt $pldadir/ivector.scp; do
        [ ! -f $f ] && echo "$0: Could not find $pldadir/spk2utt" && exit 1
      done

      $cmd $dir/plda_src_mean_tx/log/compute_plda.log \
        ivector-compute-plda \
        ark,t:$pldadir/spk2utt \
        "ark:ivector-subtract-global-mean $pldadir/mean.vec scp:$pldadir/ivector.scp ark:- | transform-vec $ivecdir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        $dir/plda_src_mean_tx/plda
      pldadir=$dir/plda_src_mean_tx
    else
      mkdir -p $dir/plda_src_mean

      cp $pldadir/transform.mat $dir/plda_src_mean
      cp $pldadir/plda $dir/plda_src_mean
      cp $ivecdir/mean.vec $dir/plda_src_mean
      pldadir=$dir/plda_src_mean
    fi
  fi
fi

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

if $do_sph_norm || $do_efr_norm; then
  num_iters=`cat $norm_dir/num_iters` || exit 1
  ivectors="$ivectors copy-vector scp:- ark:- |"
  for iter in `seq 0 $[num_iters-1]`; do
    for f in $norm_dir/mean_iter$iter.vec $norm_dir/transform_iter$iter.mat; do
      [ ! -f $f ] && echo "$0: Could not find $f" && exit 1
    done
    ivectors="$ivectors ivector-subtract-global-mean $norm_dir/mean_iter$iter.vec ark:- ark:- | transform-vec $norm_dir/transform_iter$iter.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
  done
else
  ivectors="$ivectors ivector-subtract-global-mean $pldadir/mean.vec scp:- ark:- | transform-vec $pldadir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
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
  $cmd JOB=1:$nj $dir/log/plda_scoring.JOB.log \
    ivector-plda-scoring-dense --target-energy=$target_energy $pldadir/plda \
      "$reco2utt" "$ivectors" ark,scp:$dir/scores.JOB.ark,$dir/scores.JOB.scp \
      ark,t:$dir/tmp/out_reco2utt.JOB || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining scores across jobs"
  for j in $(seq $nj); do cat $dir/scores.$j.scp; done >$dir/scores.scp || exit 1;

  if ! $per_spk; then
    for j in $(seq $nj); do cat $dir/tmp/out_reco2utt.$j; done >$dir/reco2utt || exit 1;
      utils/spk2utt_to_utt2spk.pl $dir/reco2utt > $dir/utt2reco
  else
    # out_reco2utt is really out_reco2spk
    for j in $(seq $nj); do cat $dir/tmp/out_reco2utt.$j; done >$dir/reco2spk || exit 1;
    utils/spk2utt_to_utt2spk.pl $dir/reco2spk > $dir/spk2reco
    utils/apply_map.pl -f 2 $dir/spk2reco < $dir/tmp/utt2spk > $dir/utt2reco
  fi

fi

utils/filter_scp.pl $dir/utt2reco $dir/tmp/utt2spk > $dir/utt2spk
utils/filter_scp.pl $dir/utt2reco $dir/tmp/segments > $dir/segments
utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

echo $pldadir > $dir/pldadir

#if $cleanup ; then
#  rm -rf $dir/tmp || exit 1;
#fi
