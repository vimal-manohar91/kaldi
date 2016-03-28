#!/bin/bash

# Copyright 2013  Johns Hopkins University (author: Daniel Povey)
#           2016  Vimal Manohar
# Apache 2.0

# This script operates on a directory, such as in data/train/,
# that contains some subset of the following files:
#  feats.scp
#  wav.scp
#  spk2utt
#  utt2spk
#  text
#  segments
#
# It copies to another directory and all segments 
# are padded on the left and the right by the specified amount.
# feats.scp file is deleted since it has to be extracted again.
# cmvn.scp file is also deleted. 
# If there is no segments file, this script exits with error.

# begin configuration section
pad_length_left=0
pad_length_right=0
cmd=run.pl
nj=4
# end configuration section

. utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: "
  echo "  $0 [options] <srcdir> <tmpdir> <destdir>"
  echo "e.g.:"
  echo " $0 --pad-length-left 0.02 --pad-length-right 0.04 data/train exp/pad_segments_train data/train_padded"
  exit 1;
fi

export LC_ALL=C

srcdir=$1
tmpdir=$2
destdir=$3

if [ ! -f $srcdir/utt2spk ]; then
  echo "$0: no such file $srcdir/utt2spk"
  exit 1;
fi

set -e;

utils/copy_data_dir.sh $srcdir $destdir
rm $destdir/{feats.scp,cmvn.scp} 2>/dev/null

if [ ! -f $destdir/segments ]; then
  echo "$0: no segments file found. Nothing to pad."
  exit 0
fi

utils/split_data.sh $srcdir $nj

$cmd JOB=1:$nj $tmpdir/log/get_reco_lenghts.JOB.log \
  wav-to-duration scp:$srcdir/split$nj/JOB/wav.scp ark,t:$tmpdir/reco_lengths.JOB.ark

for n in `seq $nj`; do
  cat $tmpdir/reco_lengths.$n.ark
done > $tmpdir/reco_lengths.ark

python utils/pad_segments.py --pad-length-left=$pad_length_left \
  --pad-length-right=$pad_length_right \
  --reco-lengths=$tmpdir/reco_lengths.ark \
  $srcdir/segments $destdir/segments

utils/fix_data_dir.sh $destdir
