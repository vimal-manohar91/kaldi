#!/bin/bash
# Copyright (c) 2019, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error


audio=$1
transcripts=$2

traindir=data/train
devdir=data/dev
localdir=data/local/all

mkdir -p $traindir
mkdir -p $devdir
mkdir -p $localdir

find $audio -name "*wav" -type f | \
  perl -ne ' {
    chomp;
    $wav=$_;
    s:.*/([^\/]+)\.wav:$1:;
    $wavid = sprintf("%020s", $_);
    print $wavid . " $wav\n";
  }' | sort > $localdir/wav.scp

find $transcripts -name "*json" -type f | \
  local/parse_transcripts.py > $localdir/transcripts

# `cut` does not do reordering :/
awk -F'\t' '{print $5" "$2" "$3" "$4}' $localdir/transcripts > $traindir/segments
awk -F'\t' '{print $5" "$8}' $localdir/transcripts > $traindir/text
awk -F'\t' '{print $5" "$6}' $localdir/transcripts > $traindir/utt2spk
utils/utt2spk_to_spk2utt.pl $traindir/utt2spk > $traindir/spk2utt
cat $localdir/wav.scp | grep -f local/train.lst -F -w > $traindir/wav.scp
utils/fix_data_dir.sh $traindir

# `cut` does not do reordering :/
awk -F'\t' '{print $5" "$2" "$3" "$4}' $localdir/transcripts > $devdir/segments
awk -F'\t' '{print $5" "$8}' $localdir/transcripts > $devdir/text
awk -F'\t' '{print $5" "$6}' $localdir/transcripts > $devdir/utt2spk
utils/utt2spk_to_spk2utt.pl $devdir/utt2spk > $devdir/spk2utt
cat $localdir/wav.scp | grep -f local/dev.lst -F -w > $devdir/wav.scp
utils/fix_data_dir.sh $devdir
