#!/bin/bash

# Modified based on the script: utils/data/copy_data_dir.sh 

# Copyright 2013  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script operates on a directory, such as in data/train/,
# that contains some subset of the following files:
#  feats.scp
#  wav.scp
#  spk2utt
#  utt2spk
#  text
#
# It copies to another directory, possibly adding a specified prefix or a suffix
# to the utterance and/or speaker names.  Note, the recording-ids stay the same.
#

# begin configuration section
validate_opts=   # should rarely be needed.
# end configuration section

. utils/parse_options.sh

if [ $# != 4 ]; then
  echo "Usage: "
  echo "  $0 <srcdir> <sub-segment-text> <sub-segments> <destdir>"
  echo "e.g.:"
  echo " $0 data/train data/train/sub_segment_text data/train/sub_segments data/train_1"
  exit 1;
fi

export LC_ALL=C

srcdir=$1
sub_segment_text=$2
sub_segments=$3
destdir=$4

if [ ! -f $srcdir/utt2spk ]; then
  echo "resegment_data_dir.sh: no such file $srcdir/utt2spk"
  exit 1;
fi

set -e;

rm -r $destdir 2>/dev/null || true
mkdir -p $destdir

cat $srcdir/segments | awk '{print $1, $2}' > $destdir/utt2reco_map
cat $sub_segments | awk '{print $1, $2}' > $destdir/utt_new2utt_map

if [ -f $srcdir/utt2spk ]; then
  utils/apply_map.pl -f 2 $srcdir/utt2spk < $destdir/utt_new2utt_map >$destdir/utt2spk
else
  echo "$0: no such file $srcdir/utt2spk"
  exit 1
fi

if [ -f $srcdir/utt2uniq ]; then
  utils/apply_map.pl -f 2 $srcdir/utt2uniq < $destdir/utt_new2utt_map >$destdir/utt2uniq
fi

if [ -f $sub_segments ]; then
  cat $sub_segments | awk '{print $1,($4-$3)}' > $destdir/utt2dur  
else
  echo "resegment_data_dir.sh: no such file $sub_segments"
  exit 1;
fi

if [ -f $srcdir/segments ]; then
  cp $srcdir/wav.scp $destdir
  
  python -c 'import sys
start_times = dict()
for line in open(sys.argv[1]):
  splits = line.strip().split()
  start_times[splits[0]] = float(splits[2])

for line in sys.stdin.readlines():
  splits = line.strip().split()
  splits[2] = str(float(splits[2]) + start_times[splits[1]])
  splits[3] = str(float(splits[3]) + start_times[splits[1]])
  print(" ".join(splits))' $srcdir/segments < $sub_segments | \
    utils/apply_map.pl -f 2 $destdir/utt2reco_map > $destdir/segments

  if [ -f $srcdir/reco2file_and_channel ]; then
    cp $srcdir/reco2file_and_channel $destdir/
  fi
else
  echo "resegment_data_dir.sh: no such file $srcdir/segments"
  exit 1;
fi

if [ -f $sub_segment_text ]; then
  cp $sub_segment_text $destdir/text
else
  echo "$0: no such file $sub_segment_text"
  exit 1;
fi

if [ -f $srcdir/spk2gender ]; then
  cp $srcdir/spk2gender $destdir
fi

utils/utt2spk_to_spk2utt.pl $destdir/utt2spk > $destdir/spk2utt

for f in stm glm ctm; do
  if [ -f $srcdir/$f ]; then
    cp $srcdir/$f $destdir
  fi
done

rm $destdir/utt_new2utt_map $destdir/utt2reco_map

echo "$0: resegmented data from $srcdir to $destdir"

[ ! -f $srcdir/feats.scp ] && validate_opts="$validate_opts --no-feats"
[ ! -f $srcdir/text ] && validate_opts="$validate_opts --no-text"

utils/validate_data_dir.sh $validate_opts $destdir
