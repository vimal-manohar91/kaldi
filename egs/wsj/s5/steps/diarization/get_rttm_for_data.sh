#! /bin/bash

per_utt=false

. utils/parse_options.sh
. path.sh

if [ $# -ne 1 ]; then
  echo "Usage: $0 <data>"
  echo " e.g.: $0 data/eval97.seg"
  exit 1
fi

data=$1

utils/data/get_reco2utt.sh $data

if $per_utt; then
  awk '{for (i=2; i<=NF; i++) { print i" "(i-1); }}' $data/reco2utt > $data/utt2label
else
  awk '{i++; print $1" "i}' $data/spk2utt > $data/speakers.txt
  cat $data/utt2spk | utils/sym2int.pl -f 2 $data/speakers.txt > $data/utt2label
fi

cat $data/reco2file_and_channel | \
  perl -ane 'if ($F[2] == "A") { $F[2] = "1"; } print(join(" ", @F) . "\n");' > \
  $data/reco2file_and_channel_fixed

export PATH=$PATH:$KALDI_ROOT/tools/sctk/bin
segmentation-init-from-segments --frame-overlap=0 --shift-to-zero=false \
  --utt2label-rspecifier=ark,t:$data/utt2label $data/segments ark:- | \
  segmentation-combine-segments-to-recordings ark:- ark,t:$data/reco2utt ark:- | \
  segmentation-to-rttm --map-to-speech-and-sil=false \
  --reco2file-and-channel=$data/reco2file_and_channel_fixed ark:- - | \
  rttmSmooth.pl -s 0
