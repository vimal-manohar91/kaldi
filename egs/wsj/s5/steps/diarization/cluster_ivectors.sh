#!/bin/bash

# Copyright  2016  David Snyder
# Apache 2.0.

# TODO This script performs agglomerative clustering.

# Begin configuration section.
cmd="run.pl"
stage=0
nj=10
cleanup=true
target_energy=0.1
threshold=0
reco2num_spk=
adjacency_factor=0.0
use_plda_clusterable=true
transform_plda=false
cluster_opts=
per_spk=false
use_kmeans=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <plda-dir> <ivector-dir> <output-dir>"
  echo " e.g.: $0 data/callhome exp/ivectors_callhome exp/ivectors_callhome/results"
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

rm -r $dir/tmp || true
mkdir -p $dir/tmp

echo $threshold > $dir/threshold.txt

extra_files=$ivecdir/ivector.scp
if $per_spk; then
  extra_files=$ivecdir/ivector_spk.scp
fi

do_spherical_nuisance_normalization=false
if [ -f $pldadir/snn/lda_iter0.mat ]; then
  do_spherical_nuisance_normalization=true
fi

if $do_spherical_nuisance_normalization; then
  extra_files="$extra_files $pldadir/mean.vec $pldadir/transform.mat"
fi

for f in $ivecdir/spk2utt $ivecdir/utt2spk \
  $pldadir/plda $extra_files; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

echo "$0: Preparing tmp fake data directory $dir/tmp"
cp $ivecdir/spk2utt $dir/tmp/
cp $ivecdir/utt2spk $dir/tmp/
cp $ivecdir/segments $dir/tmp/
cp $ivecdir/spk2utt $dir/
cp $ivecdir/utt2spk $dir/
cp $ivecdir/segments $dir/

utils/data/get_reco2utt.sh $dir/tmp/

utils/fix_data_dir.sh $dir/tmp > /dev/null

if [ ! -z $reco2num_spk ]; then
  reco2num_spk="ark,t:$reco2num_spk"
fi

sdata=$dir/tmp/split${nj}reco;
utils/split_data.sh --per-reco $dir/tmp $nj || exit 1;

# Set various variables.
mkdir -p $dir/log

if $per_spk; then
  utils/spk2utt_to_utt2spk.pl $dir/tmp/reco2utt > $dir/tmp/utt2reco
  utils/apply_map.pl -f 1 $dir/tmp/utt2spk < $dir/tmp/utt2reco | sort -u > $dir/tmp/spk2reco
  utils/utt2spk_to_spk2utt.pl $dir/tmp/spk2reco > $dir/tmp/reco2spk

  reco2utt="ark,t:utils/filter_scp.pl $sdata/JOB/reco2utt $dir/tmp/reco2spk |"
  if [ ! -f $ivecdir/ivector_key2samples ]; then
    ivectors="ark:utils/filter_scp.pl $sdata/JOB/spk2utt $ivecdir/ivector_spk.scp |"
  else
    for n in $(seq $nj); do 
      utils/filter_scp.pl $sdata/$n/spk2utt $ivecdir/ivector_key2samples > $sdata/$n/ivector_key2samples; 
    done
    ivectors="ark:utils/filter_scp.pl -f 2 $sdata/JOB/spk2utt $ivecdir/ivector_samples2key | utils/filter_scp.pl /dev/stdin $ivecdir/ivector_spk.scp |"
  fi
else
  reco2utt=ark,t:$sdata/JOB/reco2utt 
  if [ ! -f $ivecdir/ivector_key2samples ]; then
    ivectors="ark:utils/filter_scp.pl $sdata/JOB/utt2spk $ivecdir/ivector.scp |"
  else
    for n in $(seq $nj); do 
      utils/filter_scp.pl $sdata/$n/utt2spk $ivecdir/ivector_key2samples > $sdata/$n/ivector_key2samples; 
    done
    ivectors="ark:utils/filter_scp.pl -f 2 $sdata/JOB/utt2spk $ivecdir/ivector_samples2key | utils/filter_scp.pl /dev/stdin $ivecdir/ivector.scp |"
  fi
fi

if $do_spherical_nuisance_normalization; then
  ivectors="$ivectors copy-vector scp:- ark:- |"
  num_iters=`cat $pldadir/snn/num_iters` || exit 1
  for iter in `seq 0 $[num_iters-1]`; do
    for f in $pldadir/snn/mean_iter$iter.vec $pldadir/snn/lda_iter$iter.mat; do
      [ ! -f $f ] && echo "$0: Could not find $f" && exit 1
    done
    ivectors="$ivectors ivector-subtract-global-mean $pldadir/snn/mean_iter$iter.vec ark:- ark:- | transform-vec $pldadir/snn/lda_iter$iter.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
  done
else
  ivectors="$ivectors ivector-subtract-global-mean $pldadir/mean.vec scp:- ark:- | transform-vec $pldadir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
fi
    
if [ -f $ivecdir/ivector_key2samples ]; then
  ivectors="$ivectors pack-vectors-into-matrix ark:- ark,t:$sdata/JOB/ivector_key2samples ark:- |"
  cluster_opts="$cluster_opts --ivector-matrix-input"
fi

if $use_plda_clusterable; then
  if [ $stage -le 0 ]; then
    echo "$0: clustering scores"

    if [ $adjacency_factor != 0.0 ]; then
      echo "No longer supported!" && exit 1
      if $per_spk; then
        echo "--per-spk must be false"
        exit 1
      fi

      $cmd JOB=1:$nj $dir/log/agglomerative_cluster.JOB.log \
        agglomerative-cluster-plda-adjacency --verbose=2 --threshold=$threshold \
          --target-energy=$target_energy --adjacency-factor=$adjacency_factor \
          ${cluster_opts} \
          ${reco2num_spk:+--reco2num-spk-rspecifier="$reco2num_spk"} \
          $pldadir/plda ark,t:$sdata/JOB/reco2utt "$ivectors" \
          "ark:segmentation-init-from-segments --shift-to-zero=false --frame-overlap=0.0 $sdata/JOB/segments ark:- |" \
          ark,t:$dir/labels.JOB || exit 1;
    else
      $cmd JOB=1:$nj $dir/log/agglomerative_cluster.JOB.log \
        agglomerative-cluster-plda --verbose=2 --threshold=$threshold \
          --target-energy=$target_energy \
          ${cluster_opts} \
          ${reco2num_spk:+--reco2num-spk-rspecifier="$reco2num_spk"} \
          $pldadir/plda "$reco2utt" "$ivectors" \
          ark,t:$dir/labels.JOB ark,t:$dir/out_utt2spk.JOB || exit 1;
    fi
  fi
else
  if $transform_plda; then
    ivectors="$ivectors ivector-transform-plda --target-energy=$target_energy $pldadir/plda '$reco2utt' ark:- ark:- |"
  fi

  if ! $use_kmeans; then
    if [ $adjacency_factor != 0.0 ]; then
      if $per_spk; then
        echo "--per-spk must be false"
        exit 1
      fi

      if [ $stage -le 0 ]; then
        echo "$0: clustering scores"
        $cmd JOB=1:$nj $dir/log/agglomerative_cluster.JOB.log \
          agglomerative-cluster-vector-adjacency --verbose=2 --threshold=$threshold --target-energy=$target_energy \
            --adjacency-factor=$adjacency_factor \
            ${reco2num_spk:+--reco2num-spk-rspecifier="$reco2num_spk"} \
            "$reco2utt" "$ivectors" \
            "ark:segmentation-init-from-segments --frame-overlap=0.0 --shift-to-zero=false $sdata/JOB/segments ark:- |" \
            ark,t:$dir/labels.JOB || exit 1;
      fi
    else
      if [ $stage -le 0 ]; then
        echo "$0: clustering scores"
        $cmd JOB=1:$nj $dir/log/agglomerative_cluster.JOB.log \
          agglomerative-cluster-vectors --verbose=2 --threshold=$threshold \
            ${reco2num_spk:+--reco2num-spk-rspecifier="$reco2num_spk"} \
            "$reco2utt" "$ivectors" \
            ark,t:$dir/labels.JOB ark,t:$dir/out_utt2spk.JOB || exit 1;
      fi
    fi
  else
    if [ $stage -le 0 ]; then
      echo "$0: clustering scores"
      $cmd JOB=1:$nj $dir/log/kmeans_cluster.JOB.log \
        kmeans-cluster-vectors --verbose=2 \
        ${reco2num_spk:+--reco2num-spk-rspecifier="$reco2num_spk"} \
        "$reco2utt" "$ivectors" \
        ark,t:$dir/labels.JOB ark,t:$dir/out_utt2spk.JOB || exit 1;
    fi
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
    utils/apply_map.pl -f 2 $dir/labels_spk < $dir/tmp/utt2spk > $dir/labels || exit 1
    utils/apply_map.pl -f 2 $dir/out_spk2cluster < $dir/tmp/utt2spk > $dir/out_utt2spk || exit 1
  else
    for j in $(seq $nj); do cat $dir/labels.$j; done > $dir/labels || exit 1;
    for j in $(seq $nj); do cat $dir/out_utt2spk.$j; done > $dir/out_utt2spk || exit 1;
  fi
fi

if [ $stage -le 2 ]; then
  echo "$0: computing RTTM"
  cat $ivecdir/segments | sort -k2,2 -k3,4n | \
    python steps/diarization/make_rttm.py /dev/stdin $dir/labels > $dir/rttm || exit 1;
fi

if $cleanup ; then
  rm -rf $dir/tmp || exit 1;
fi

