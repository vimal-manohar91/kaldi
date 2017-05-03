#! /bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0.

# Prepares data directory with oracle speakers

dataset=train
mic=mdm8

. path.sh

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0 --dataset <dataset> --mic <mic-id>"
  echo " e.g.: $0 --dataset train --mic mdm8"
  exit 1
fi

for f in data/$mic/${dataset}_orig/{segments,utt2spk,text} \
  data/ihm/${dataset}_orig/{segments,utt2spk,text}; do
  [ ! -f $f ] && echo "$0: Could not find file $f" && exit 1
done

if [ "$mic" == "ihm" ]; then
  utils/copy_data_dir.sh ${dataset}_orig ${dataset}_oraclespk
fi

get_utt2reco() {
  data=$1
  awk '{print $1" "$2}' $data/segments > $data/utt2reco
}

local/prepare_parallel_train_data.sh --train-set ${dataset}_orig $mic
get_utt2reco data/ihm/${dataset}_orig
get_utt2reco data/$mic/${dataset}_orig
    
cat data/$mic/${dataset}_orig_ihmdata/ihmutt2utt | \
  utils/apply_map.pl -f 1 data/ihm/${dataset}_orig/utt2reco | \
  utils/apply_map.pl -f 2 data/$mic/${dataset}_orig/utt2reco | \
  sort -u > data/$mic/${dataset}_orig_ihmdata/ihm2${mic}_reco

utils/copy_data_dir.sh data/$mic/${dataset}_orig \
  data/$mic/${dataset}_oraclespk

cat data/ihm/${dataset}_orig/utt2spk | \
  utils/filter_scp.pl data/$mic/${dataset}_orig_ihmdata/ihmutt2utt | \
  utils/apply_map.pl -f 1 data/$mic/${dataset}_orig_ihmdata/ihmutt2utt > \
  data/$mic/${dataset}_oraclespk/utt2spk.temp
    
cat data/$mic/${dataset}_oraclespk/utt2spk.temp | \
  awk '{print $1" "$2"-"$1}' > \
  data/$mic/${dataset}_oraclespk/utt2newutt
  
utils/apply_map.pl -f 1 data/$mic/${dataset}_oraclespk/utt2newutt \
  < data/$mic/${dataset}_oraclespk/utt2spk.temp > \
  data/$mic/${dataset}_oraclespk/utt2spk
    
for f in segments text; do
  utils/apply_map.pl -f 1 data/$mic/${dataset}_oraclespk/utt2newutt \
    < data/$mic/${dataset}_orig/$f > \
    data/$mic/${dataset}_oraclespk/$f
done
     
rm data/$mic/${dataset}_oraclespk/{feats.scp,spk2utt,cmvn.scp} || true

utils/fix_data_dir.sh data/$mic/${dataset}_oraclespk
