#!/bin/bash

# Copyright 2012-2016  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Compute cepstral mean and variance statistics per utterance.
# We do this in just one job; it's fast.
# This script takes no options.

echo "$0 $@"  # Print the command line for logging

fake=false   # If specified, can generate fake/dummy CMVN stats (that won't normalize)
fake_dims=   # as the "fake" option, but you can generate "fake" stats only for certain
             # dimensions.
two_channel=false

if [ "$1" == "--fake" ]; then
  fake=true
  shift
fi
if [ "$1" == "--fake-dims" ]; then
  fake_dims=$2
  shift
  shift
fi
if [ "$1" == "--two-channel" ]; then
  two_channel=true
  shift
fi

cmd=run.pl
nj=1

. utils/parse_options.sh

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
   echo "Usage: $0 [options] <data-dir> [<log-dir> [<cmvn-dir>] ]";
   echo "e.g.: $0 data/train exp/make_mfcc/train mfcc"
   echo "Note: <log-dir> defaults to <data-dir>/log, and <cmvn-dir> defaults to <data-dir>/data"
   echo "Options:"
   echo " --fake          gives you fake cmvn stats that do no normalization."
   echo " --two-channel   is for two-channel telephone data, there must be no segments "
   echo "                 file and reco2file_and_channel must be present.  It will take"
   echo "                 only frames that are louder than the other channel."
   echo " --fake-dims <n1:n2>  Generate stats that won't cause normalization for these"
   echo "                  dimensions (e.g. 13:14:15)"
   exit 1;
fi

if [ -f path.sh ]; then . ./path.sh; fi

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=$data/log
fi
if [ $# -ge 3 ]; then
  cmvndir=$3
else
  cmvndir=$data/data
fi

# make $cmvndir an absolute pathname.
cmvndir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $cmvndir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $cmvndir || exit 1;
mkdir -p $logdir || exit 1;


required="$data/feats.scp"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_cmvn.sh: no such file $f"
    exit 1;
  fi
done

if $fake; then
  dim=`feat-to-dim scp:$data/feats.scp -`
  ! cat $data/utt2spk | awk -v dim=$dim '{print $1, "["; for (n=0; n < dim; n++) { printf("0 "); } print "1";
                                                        for (n=0; n < dim; n++) { printf("1 "); } print "0 ]";}' | \
    copy-matrix ark:- ark,scp:$cmvndir/cmvn_perutt_$name.ark,$cmvndir/cmvn_perutt_$name.scp && \
     echo "Error creating fake CMVN stats" && exit 1;
elif $two_channel; then
  ! compute-cmvn-stats-two-channel $data/reco2file_and_channel scp:$data/feats.scp \
       ark,scp:$cmvndir/cmvn_perutt_$name.ark,$cmvndir/cmvn_perutt_$name.scp \
    2> $logdir/cmvn_perutt_$name.log && echo "Error computing CMVN stats (using two-channel method)" && exit 1;
elif [ ! -z "$fake_dims" ]; then
  ! compute-cmvn-stats scp:$data/feats.scp ark:- | \
    modify-cmvn-stats "$fake_dims" ark:- ark,scp:$cmvndir/cmvn_perutt_$name.ark,$cmvndir/cmvn_perutt_$name.scp && \
    echo "Error computing (partially fake) CMVN stats" && exit 1;
else
  utils/split_data.sh --per-utt $data $nj
  sdata=${data}/split${nj}utt
  $cmd JOB=1:$nj $logdir/cmvn_perutt_$name.JOB.log \
    compute-cmvn-stats scp:$sdata/JOB/feats.scp ark,scp:$cmvndir/cmvn_perutt_$name.JOB.ark,$cmvndir/cmvn_perutt_$name.JOB.scp || \
    { echo "Error computing CMVN stats" && exit 1; }
   
  for n in `seq $nj`; do
    cat $cmvndir/cmvn_perutt_$name.$n.scp
  done > $cmvndir/cmvn_perutt_$name.scp
fi

cp $cmvndir/cmvn_perutt_$name.scp $data/cmvn_perutt.scp || exit 1;

nc=`cat $data/cmvn_perutt.scp | wc -l`
nu=`cat $data/utt2spk | wc -l`
if [ $nc -ne $nu ]; then
  echo "$0: warning: it seems not all of the speakers got cmvn stats ($nc != $nu);"
  [ $nc -eq 0 ] && exit 1;
fi

echo "Succeeded creating CMVN stats for $name"

