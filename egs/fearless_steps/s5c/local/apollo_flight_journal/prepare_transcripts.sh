#!/bin/bash

dict=data/local/dict
afj_output=/export/fs01/vmanoha1/fearless_steps/apollo_journal/output_lines_v1
dir=data/local/afj
missions="7 8 9 10 12 13 14 15 16 17"
stage=0

. utils/parse_options.sh

mkdir -p $dir

if [ $stage -le 0 ]; then
  for m in $missions; do
    local/apollo_flight_journal/parse_afj_transcripts.py \
      $afj_output/apollo${m}_transcripts.jl \
      $dir/apollo${m}_transcripts.txt
  done
fi

if [ $stage -le 1 ]; then
  for m in $missions; do
    cat $dir/apollo${m}_transcripts.txt | local/dict/normalize_abbrv.py $dict/lexicon.txt
  done > $dir/afj_transcripts.txt
fi
