#!/bin/bash

if [ $# -ne 3 ]; then
  echo "Usage: $0 <data> <map> <dir>"
  exit 1
fi

data=$1
map=$2
dir=$3

utils/copy_data_dir.sh $data $dir

for f in utt2spk segments feats.scp; do
  if [ -f $data/$f ]; then
    utils/apply_map.pl -f 1 $map < $data/$f > $dir/$f
  fi
done

rm $dir/spk2utt

utils/fix_data_dir.sh $dir
