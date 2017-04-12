#! /bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <data>"
  exit 1
fi

data=$1

utils/data/get_reco2utt.sh $data
awk '{for (i=2; i<=NF; i++) { print $i" "i-1; }}' $data/reco2utt > $data/utterances.txt

sort -k3,4 -n $data/segments | \
  steps/diarization/make_rttm.py --reco2file-and-channel=$data/reco2file_and_channel - $data/utt2spk > $data/dummy_rttm

md-eval.pl -s $data/rttm -r $data/dummy_rttm -c 0 -o -M $data/mapping.txt

cut -d ' ' -f 2 $data/utt2spk | \
  steps/diarization/get_best_mapping.py --ref-speakers - \
  $data/mapping.txt > $data/best_mapping.txt

cat $data/utt2spk | \
  utils/apply_map.pl -f 2 $data/best_mapping.txt > $data/utt2ref_spk
