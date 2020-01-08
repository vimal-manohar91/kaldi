#!/bin/bash

# Copyright 2018
# Apache 2.0

stage=0
cmd=run.pl
acwt=0.1
beam=8.0
min_prob=0.01
min_relative_prob=0.01
word_determinize=true
write_compact=true

. utils/parse_options.sh
. ./path.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 <latdir> <dir>"
  exit 1
fi

latdir=$1
dir=$2

nj=$(cat $latdir/num_jobs) || exit 1

srcdir=$latdir
if [ ! -f $srcdir/final.mdl ]; then
  srcdir=$(dirname $latdir)
  if [ ! -f $srcdir/final.mdl ]; then
    echo "$0: Could not find $latdir/final.mdl or $latdir/../final.mdl"
    exit 1
  fi
fi

if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/mbr_decode.JOB.log \
    lattice-push "ark:gunzip -c $latdir/lat.JOB.gz |" ark:- \| \
    lattice-mbr-decode --acoustic-scale=$acwt ark:- ark:/dev/null ark:/dev/null \
    "ark:| gzip -c > $dir/sausage.JOB.gz"
fi

if [ $stage -le 2 ]; then
  $cmd JOB=1:$nj $dir/log/make_graphs.JOB.log \
    copy-post "ark:gunzip -c $dir/sausage.JOB.gz |" ark,t:- \| \
    utils/sausage_to_G_fst_simple.py --write-lattice=true --min-prob=$min_prob --min-relative-prob=$min_relative_prob \| \
    lattice-scale --acoustic-scale=0.0 --lm-scale=0.0 ark,t:- "ark:| gzip -c > $dir/sausage_lat.JOB.gz" || exit 1
fi

if [ $stage -le 3 ]; then
  $cmd JOB=1:$nj $dir/log/posterior_prune_lat.JOB.log \
    lattice-compose --write-compact=$write_compact "ark:gunzip -c $latdir/lat.JOB.gz |" \
      "ark:gunzip -c $dir/sausage_lat.JOB.gz |" ark:- \| \
    lattice-determinize-phone-pruned --acoustic-scale=$acwt --beam=$beam \
      --write-compact=$write_compact --word-determinize=$word_determinize \
      $srcdir/final.mdl ark:- "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1
fi

echo $nj > $dir/num_jobs

exit 0
