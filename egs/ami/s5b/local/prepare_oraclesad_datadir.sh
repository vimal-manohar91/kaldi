#! /bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0.

dataset=dev
mic=mdm8

. path.sh
. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0 --dataset <dataset> --mic <mic-id>"
  echo " e.g.: $0 --dataset dev --mic mdm8"
  exit 1
fi

for f in segments wav.scp reco2file_and_channel; do
  if [ ! -f data/$mic/${dataset}/$f ]; then
   echo "$0: Could not find file data/$mic/${dataset}/$f" && exit 1
 fi
done

rm -r data/$mic/${dataset}_oraclesad || true
mkdir -p data/$mic/${dataset}_oraclesad

utils/data/get_reco2utt.sh data/$mic/${dataset}
segmentation-init-from-segments --shift-to-zero=false data/$mic/${dataset}/segments ark:- | \
  segmentation-combine-segments-to-recordings ark:- ark,t:data/$mic/${dataset}/reco2utt ark:- | \
  segmentation-post-process --merge-adjacent-segments ark:- ark:- | \
  segmentation-to-segments --single-speaker ark:- \
  ark,t:data/$mic/${dataset}_oraclesad/utt2spk data/$mic/${dataset}_oraclesad/segments

cp data/$mic/${dataset}/{wav.scp,reco2file_and_channel} data/$mic/${dataset}_oraclesad
utils/fix_data_dir.sh data/$mic/${dataset}_oraclesad
