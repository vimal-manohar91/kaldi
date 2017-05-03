#! /bin/bash

per_spk=true

. utils/parse_options.sh 

if [ $# -ne 1 ]; then
  echo "Usage: $0 <data>"
  exit 1
fi

data=$1

mkdir -p $data/tmp

utils/data/get_reco2utt.sh $data
cat $data/reco2file_and_channel | perl -ane 'if ($F[2] == "A") { $F[2] = "1"; }; print join(" ", @F) . "\n";' >  $data/tmp/new_reco2file_and_channel

if $per_spk; then
  #sort -k3,4 -n $data/segments | \
  #  steps/diarization/make_rttm.py --reco2file-and-channel=$data/new_reco2file_and_channel - $data/utt2spk > $data/dummy_rttm

  awk '{i++; print $1" "i}' $data/spk2utt > $data/tmp/speakers.txt

  segmentation-init-from-segments --frame-overlap=0 --shift-to-zero=false \
    --utt2label-rspecifier="ark,t:utils/sym2int.pl -f 2 $data/tmp/speakers.txt $data/utt2spk |" \
    $data/segments ark:- | \
    segmentation-combine-segments-to-recordings ark:- ark,t:$data/reco2utt ark:- | \
    segmentation-to-rttm --reco2file-and-channel=$data/tmp/new_reco2file_and_channel \
    --map-to-speech-and-sil=false ark:- $data/tmp/dummy_rttm

  md-eval.pl -s $data/rttm -r $data/tmp/dummy_rttm -c 0 -o -M $data/tmp/mapping.txt

  cut -d ' ' -f 2 $data/utt2spk | \
    steps/diarization/get_best_mapping.py --ref-speakers - --write-overlapping-info=$data/tmp/spk2overlapping_fraction \
    $data/tmp/mapping.txt > $data/tmp/best_mapping.txt

  cp $data/tmp/best_mapping.txt $data/tmp/spk2best_spk
  cat $data/utt2spk | \
    utils/apply_map.pl -f 2 $data/tmp/best_mapping.txt > $data/utt2ref_spk
  utils/int2sym.pl -f 1 $data/tmp/speakers.txt $data/tmp/spk2overlapping_fraction > $data/spk2overlapping_fraction
else
  # awk '{for (i=2; i<=NF; i++) { print $i" "$i"-"(i-1); }}' $data/reco2utt > $data/utterances.txt
  awk '{i++; print $1" "i}' $data/utt2spk > $data/tmp/utterances.txt

  #sort -k3,4 -n $data/segments | \
  #  steps/diarization/make_rttm.py --reco2file-and-channel=$data/new_reco2file_and_channel - \
  #  $data/utterances.txt > $data/dummy_rttm

  segmentation-init-from-segments --frame-overlap=0 --shift-to-zero=false \
    --utt2label-rspecifier=ark,t:$data/tmp/utterances.txt \
    $data/segments ark:- | \
    segmentation-combine-segments-to-recordings ark:- ark,t:$data/reco2utt ark:- | \
    segmentation-to-rttm --reco2file-and-channel=$data/tmp/new_reco2file_and_channel \
    --map-to-speech-and-sil=false ark:- $data/tmp/dummy_rttm

  md-eval.pl -s $data/rttm -r $data/tmp/dummy_rttm -c 0 -o -M $data/tmp/mapping.txt

  cut -d ' ' -f 2 $data/tmp/utterances.txt | \
    steps/diarization/get_best_mapping.py --ref-speakers - --write-overlapping-info=$data/tmp/utt2overlapping_fraction \
    $data/tmp/mapping.txt > $data/tmp/best_mapping.txt

  utils/int2sym.pl -f 1 $data/tmp/utterances.txt $data/tmp/utt2overlapping_fraction > $data/utt2overlapping_fraction
  cat $data/tmp/utterances.txt | \
    utils/apply_map.pl -f 2 $data/tmp/best_mapping.txt > $data/utt2ref_spk
fi
