#!/bin/bash
# Copyright (c) 2019, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
#                     Vimal Manohar
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error


nj=4
cmd=run.pl

. path.sh
. utils/parse_options.sh

if [ $# -ne 1 ]; then
  echo "Usage: $0 <full-corpus-dir>"
  echo " e.g.: $0 /export/jtrmal/fs01/media"
  exit 1
fi

FULL_CORPUS_DIR=$1

localdir=data/local/apollo11
traindir=data/apollo11

mkdir -p $traindir
mkdir -p $localdir

if [ ! -f $localdir/wav.scp ]; then
  find $FULL_CORPUS_DIR -name "*.wav" -type f |
    perl -ne ' {
      chomp;
      $wav=$_;
      s:.*/([^\/]+)\.wav:$1:;
      $wavid = sprintf("%020s", $_);
      print $wavid . " $wav\n";
    }' | sort > $localdir/wav.scp
fi

find $FULL_CORPUS_DIR -name "*.tra" -type f > $localdir/transcript_list

splits=
for n in $(seq $nj); do
  splits="$splits $localdir/transcript_list.$n"
done

utils/split_scp.pl $localdir/transcript_list $splits

$cmd JOB=1:$nj $localdir/parse_transcripts.JOB.log \
  local/apollo11/parse_transcripts.py '<' $localdir/transcript_list.JOB '>' \
    $localdir/transcripts.JOB || exit 1

for n in $(seq $nj); do
  cat $localdir/transcripts.$n
done > $localdir/transcripts

# `cut` does not do reordering :/
awk -F'\t' '{print $5" "$2" "$3" "$4}' $localdir/transcripts > $traindir/segments
awk -F'\t' '{print $5" "$8}' $localdir/transcripts > $traindir/text
awk -F'\t' '{print $5" "$6}' $localdir/transcripts > $traindir/utt2spk
utils/utt2spk_to_spk2utt.pl $traindir/utt2spk > $traindir/spk2utt
cp $localdir/wav.scp $traindir/wav.scp
utils/fix_data_dir.sh $traindir

local/apollo11/convert_data_dir_to_whole.sh $traindir ${traindir}_whole

local/apollo11/select_legal_data_dir.sh ${traindir}_whole ${traindir}_whole_legal
