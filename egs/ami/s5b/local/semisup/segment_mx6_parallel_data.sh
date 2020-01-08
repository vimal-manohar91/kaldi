#!/bin/bash

set -euo pipefail

src_seg_data=data/mx6_mic_02_1a_seg
tgt_data=data/mx6_mic_04_to_13_4k

tgt_seg_data=data/mx6_mic_04_to_13_4k_1a_seg

mkdir -p $tgt_seg_data

utils/fix_data_dir.sh $src_seg_data

cut -d ' ' -f 1 $src_seg_data/utt2spk | \
  perl -ne 'm/^(\S+)_02(\S*)$/; print $1 . " " . $2 . "\n";' > \
  $tgt_seg_data/utt_root

cut -d ' ' -f 2 $src_seg_data/segments | \
  perl -ne 'm/^(\S+)_02(\S*)$/; print $1 . " " . $2 . "\n";' > \
  $tgt_seg_data/reco_root

for mic in 04 05 06 07 08 09 10 11 12 13 14; do
  cat $tgt_seg_data/utt_root | \
    awk -v mic=$mic '{print $1"_"mic$2" "$1"_02"$2}'
done | sort -k1,1 > $tgt_seg_data/utt_map

for mic in 04 05 06 07 08 09 10 11 12 13 14; do
  cat $tgt_seg_data/reco_root | \
    awk -v mic=$mic '{print $1"_"mic$2}'
done | sort -k1,1 > $tgt_seg_data/reco_list

paste -d ' ' <(cut -d ' ' -f 1 $tgt_seg_data/utt_map) $tgt_seg_data/reco_list \
  > $tgt_seg_data/utt2reco

for f in utt2spk text utt2dur utt2num_frames; do
  if [ -f $src_seg_data/$f ]; then
    utils/apply_map.pl -f 2 $src_seg_data/$f < $tgt_seg_data/utt_map > \
      $tgt_seg_data/$f
  fi
done

cp $tgt_data/wav.scp $tgt_seg_data/wav.scp

cat $tgt_seg_data/utt_map | utils/apply_map.pl -f 2 $src_seg_data/segments | \
  awk '{print $1" "$1" "$3" "$4}' | utils/apply_map.pl -f 2 $tgt_seg_data/utt2reco > \
  $tgt_seg_data/segments

utils/fix_data_dir.sh $tgt_seg_data
