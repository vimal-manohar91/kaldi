#!/bin/bash

srcdir=$1
destdir=$2

if [ $# -ne 2 ]; then
  echo "Usage: $0 <src-dir> <dest-dir>"
  exit 1
fi

utils/copy_data_dir.sh $srcdir $destdir

cat $srcdir/utt2spk | perl -ne 'm/((\S+)(_SDM\S+))\s+(\S+)/; print "$1 $4$3\n"' > $destdir/utt_map

cat $srcdir/utt2spk | utils/apply_map.pl -f 1 $destdir/utt_map >$destdir/utt2spk

if [ -f $srcdir/feats.scp ]; then
  utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/feats.scp >$destdir/feats.scp
fi

if [ -f $srcdir/vad.scp ]; then
  utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/vad.scp >$destdir/vad.scp
fi

if [ -f $srcdir/segments ]; then
  utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/segments >$destdir/segments
  cp $srcdir/wav.scp $destdir
else # no segments->wav indexed by utt.
  if [ -f $srcdir/wav.scp ]; then
    utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/wav.scp >$destdir/wav.scp
  fi
fi

if [ -f $srcdir/text ]; then
  utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/text >$destdir/text
fi

if [ -f $srcdir/utt2dur ]; then
  utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/utt2dur >$destdir/utt2dur
fi

if [ -f $srcdir/reco2dur ]; then
  if [ -f $srcdir/segments ]; then
    cp $srcdir/reco2dur $destdir/reco2dur
  else
    utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/reco2dur >$destdir/reco2dur
  fi
fi

utils/fix_data_dir.sh $destdir
