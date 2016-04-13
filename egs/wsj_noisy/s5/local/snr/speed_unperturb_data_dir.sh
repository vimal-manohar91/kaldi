#!/bin/bash

. ./path.sh

data=$1
tmpdir=$2
dir=$3

if [ $# -ne 3 ]; then
  echo "Usage: $0 <data> <tmpdir> <dir>"
  echo " e.g.: $0 data/train_sp exp/unperturb_speed_train data/train"
  exit 1
fi

mkdir -p $tmpdir

cat $data/utt2spk | awk '/sp1.0-/{print $1}' > $tmpdir/utt_list
cat $data/spk2utt | awk '/sp1.0-/{print $1}' > $tmpdir/spk_list

if [ -f $data/segmemts ]; then
  cat $data/segments | awk '/sp1.0-/{print $2}' > $tmpdir/reco_list
else
  cat $data/wav.scp | awk '/sp1.0-/{print $1}' > $tmpdir/reco_list
fi

utils/subset_data_dir.sh --utt-list $tmpdir/utt_list $data $tmpdir/data_tmp

perl -ne 'm/sp1.0-(.+)/; print "sp1.0-$1 $1\n"' $tmpdir/utt_list > $tmpdir/utt_map
perl -ne 'm/sp1.0-(.+)/; print "sp1.0-$1 $1\n"' $tmpdir/spk_list > $tmpdir/spk_map
perl -ne 'm/sp1.0-(.+)/; print "sp1.0-$1 $1\n"' $tmpdir/reco_list > $tmpdir/reco_map

utils/copy_data_dir.sh $tmpdir/data_tmp $dir
cat $tmpdir/data_tmp/utt2spk | utils/apply_map.pl -f 1 $tmpdir/utt_map | utils/apply_map.pl -f 2 $tmpdir/spk_map > $dir/utt2spk
cat $tmpdir/data_tmp/wav.scp | utils/apply_map.pl -f 1 $tmpdir/reco_map > $dir/wav.scp

if [ -f $data/segments ]; then
  cat $tmpdir/data_tmp/segments | utils/apply_map.pl -f 1 $tmpdir/utt_map | utils/apply_map.pl -f 2 $tmpdir/reco_map > $dir/segments
fi

for f in feats.scp text; do
  if [ -f $data/$f ]; then
    cat $tmpdir/data_tmp/$f | utils/apply_map.pl -f 1 $tmpdir/utt_map > $dir/$f
  fi
done

for f in cmvn.scp; do
  if [ -f $data/$f ]; then
    cat $tmpdir/data_tmp/$f | utils/apply_map.pl -f 1 $tmpdir/spk_map > $dir/$f
  fi
done

if [ -f $data/reco2file_and_channel ]; then
  cat $tmpdir/data_tmp/reco2file_and_channel | utils/apply_map.pl -f 1,2 $tmpdir/reco_map > $dir/reco2file_and_channel
fi

utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

