#!/bin/bash

# Copyright  2016  David Snyder
#            2017  Vimal Manohar
# Apache 2.0.

# This script performs agglomerative clustering using matrix of pairwise
# scores (not distances).

# Begin configuration section.
cmd="run.pl"
stage=0
nj=10
cleanup=true
threshold=0.0     # Threshold on distances (not scores).
                  # Clusters are merged if they are closer than this 
                  # distance threshold.
reco2num_spk=
compartment_size=0
adjacency_factor=0.0
cluster_opts=
per_spk=false
use_kmeans=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <src-dir> <dir>"
  echo " e.g.: $0 exp/ivectors_callhome exp/ivectors_callhome/results"
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

srcdir=$1
dir=$2

mkdir -p $dir/tmp

echo $threshold > $dir/threshold.txt

for f in $srcdir/scores.scp $srcdir/spk2utt $srcdir/utt2spk $srcdir/segments ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

cp $srcdir/spk2utt $dir/tmp/
cp $srcdir/utt2spk $dir/tmp/
cp $srcdir/segments $dir/tmp/

utils/data/get_reco2utt.sh $dir/tmp/
cp $dir/tmp/reco2utt $dir

utils/spk2utt_to_utt2spk.pl $dir/tmp/reco2utt > $dir/tmp/utt2reco
utils/apply_map.pl -f 1 $dir/tmp/utt2spk < $dir/tmp/utt2reco | sort -u > $dir/tmp/spk2reco
utils/utt2spk_to_spk2utt.pl $dir/tmp/spk2reco > $dir/tmp/reco2spk

utils/fix_data_dir.sh $dir/tmp > /dev/null

if [ ! -z "$reco2num_spk" ]; then
  reco2num_spk="ark,t:$reco2num_spk"
fi

sdata=$dir/tmp/split${nj}reco
utils/split_data.sh --per-reco $dir/tmp $nj || exit 1;

# Set various variables.
mkdir -p $dir/log

if $per_spk; then
  reco2utt="ark,t:utils/filter_scp.pl $sdata/JOB/reco2utt $dir/tmp/reco2spk |"
else
  reco2utt=ark,t:$sdata/JOB/reco2utt 
fi

feats="scp:utils/filter_scp.pl $sdata/JOB/reco2utt $srcdir/scores.scp |"
if [ $stage -le 0 ]; then
  echo "$0: clustering scores"
  if ! $use_kmeans; then
    if [ $adjacency_factor != 0.0 ]; then
      $cmd JOB=1:$nj $dir/log/agglomerative_cluster.JOB.log \
          agglomerative-group-cluster-adjacency --verbose=3 --threshold=$threshold \
            --compartment-size=$compartment_size \
            --adjacency-factor=$adjacency_factor $cluster_opts \
            ${reco2num_spk:+--reco2num-spk-rspecifier="$reco2num_spk"} "$feats" \
            "$reco2utt" \
            "ark:segmentation-init-from-segments --shift-to-zero=false --frame-overlap=0.0 $sdata/JOB/segments ark:- |" \
            ark,t:$dir/labels.JOB || exit 1;
    else
      if [ $compartment_size -gt 0 ]; then
        $cmd JOB=1:$nj $dir/log/agglomerative_cluster.JOB.log \
          agglomerative-group-cluster --verbose=3 --threshold=$threshold \
            --compartment-size=$compartment_size $cluster_opts \
            ${reco2num_spk:+--reco2num-spk-rspecifier="$reco2num_spk"} "$feats" \
            "$reco2utt" ark,t:$dir/labels.JOB ark,t:$dir/out_utt2spk.JOB || exit 1;
      else
        $cmd JOB=1:$nj $dir/log/agglomerative_cluster.JOB.log \
          agglomerative-cluster --verbose=3 --threshold=$threshold $cluster_opts \
            ${reco2num_spk:+--reco2num-spk-rspecifier="$reco2num_spk"} "$feats" \
            "$reco2utt" ark,t:$dir/labels.JOB ark,t:$dir/out_utt2spk.JOB || exit 1;
      fi
    fi
  else
    $cmd JOB=1:$nj $dir/log/kmeans_cluster.JOB.log \
      kmeans-cluster --verbose=3 $cluster_opts --apply-sigmoid=false \
      ${reco2num_spk:+--reco2num-spk-rspecifier="$reco2num_spk"} "$feats" \
      "$reco2utt" ark,t:$dir/labels.JOB ark,t:$dir/out_utt2spk.JOB || exit 1;
  fi
fi

if [ $stage -le 1 ]; then
  echo "$0: combining labels"
  if $per_spk; then
    for j in $(seq $nj); do 
      cat $dir/labels.$j; 
    done > $dir/labels_spk || exit 1;
    for j in $(seq $nj); do 
      cat $dir/out_utt2spk.$j;
    done > $dir/out_spk2cluster
    utils/apply_map.pl -f 2 $dir/labels_spk < $dir/tmp/utt2spk > $dir/labels
    utils/apply_map.pl -f 2 $dir/out_spk2cluster < $dir/tmp/utt2spk > $dir/out_utt2spk
  else
    for j in $(seq $nj); do cat $dir/labels.$j; done > $dir/labels || exit 1;
    for j in $(seq $nj); do cat $dir/out_utt2spk.$j; done > $dir/out_utt2spk || exit 1;
  fi
fi

if [ $stage -le 2 ]; then
  echo "$0: computing RTTM"
  cat $srcdir/segments | sort -k2,2 -k3,4n | \
    python steps/diarization/make_rttm.py /dev/stdin $dir/labels > $dir/rttm || exit 1;
fi

if $cleanup ; then
  rm -rf $dir/tmp || exit 1;
fi
