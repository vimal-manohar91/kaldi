#!/bin/bash
# Copyright (c) 2018, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
stage=0
# End configuration section
. ./utils/parse_options.sh

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

echo "$0 $@"  # Print the command line for logging

if [ $# -ne 1 ]; then
  echo "Wrong number of command line parameters!"
  echo "Usage: $0 <corpus_path>"
  echo " e.g.: $0 ./corpus/"
  exit 1
fi
corpus=$1

mkdir -p data/local/
mkdir -p data/train
mkdir -p data/dev

if [ $stage -le 0 ] ; then
  find $corpus -name "*flac"   > data/local/audio.lst
  grep -F \
    -f <(sed -e 's/.flac//g' -e 's/.*\///g' data/local/audio.lst) \
    $corpus/utt_spk_text.tsv > data/local/utt_spk_text.tsv
  grep -w -F -f local/dev_utt.lst data/local/utt_spk_text.tsv | \
    sort > data/dev/utt_spk_text.tsv
  grep -v -w -F -f local/dev_utt.lst data/local/utt_spk_text.tsv | \
    sort > data/train/utt_spk_text.tsv
fi

if [ $stage -le 1 ] ; then
  for dataset in train dev; do
    awk -F  $'\t' '{print $2 "_" $1, $2;}' data/${dataset}/utt_spk_text.tsv | sort \
      > data/${dataset}/utt2spk
    awk -F  $'\t' '{print $2 "_" $1, $3;}' \
      data/${dataset}/utt_spk_text.tsv | sort > data/${dataset}/text
    grep -F -f <(cut -f 1 data/${dataset}/utt_spk_text.tsv)  \
      data/local/audio.lst > data/${dataset}/audio.lst
    utils/utt2spk_to_spk2utt.pl < data/${dataset}/utt2spk \
      > data/${dataset}/spk2utt
  done
fi


if [ $stage -le 2 ] ; then
  for dataset in train dev; do
    perl -e '
      %SPK;
      open(UTTSPK, $ARGV[0]) or die "Cannot open $ARGV[0]";
      while (<UTTSPK>) {
        @fields = split /\t/;
        $SPK{$fields[0]} = $fields[1];
      }
      while (<STDIN>) {
        chomp;
        ($key = $_ ) =~ s/.*\/([^\/]+)\.flac/$1/g;
        $spk = $SPK{$key};
        print "${spk}_${key} sox $_ -t wav -r 16000 -|\n";
      }
    ' data/${dataset}/utt_spk_text.tsv \
      < data/${dataset}/audio.lst |\
      sort > data/$dataset/wav.scp
    utils/fix_data_dir.sh data/$dataset
  done
fi

if [ $stage -le 3 ] ; then
  awk -F  $'\t' '{print $2 "_" $1, $3;}' $corpus/utt_spk_text.tsv | sort |\
    grep -v -w -F -f <(awk '{print $1}' data/{dev,train}/text | sort -u) \
    > data/local/extra_text
fi

