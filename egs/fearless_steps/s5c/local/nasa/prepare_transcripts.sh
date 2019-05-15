#!/bin/bash

dict=data/local/dict_afj

afj_missions_dir=/export/fs01/vmanoha1/fearless_steps/apollo_journal/afj_transcript_lines_v1
afj_missions="7 8 9 10 12 13 14 15 16 17"

spacelog_missions_dir=/export/fs01/vmanoha1/fearless_steps/Spacelog/missions
spacelog_missions="g3 g4 g6 g7 g8 ma6 ma7 ma8 mr3 mr4"

alsj_missions_dir=/export/fs01/vmanoha1/fearless_steps/apollo_journal/alsj_transcript_lines_v1
alsj_missions="a12 a13 a14 a15 a16 a17"

apollo_html_reports=/export/fs01/vmanoha1/fearless_steps/apollo_journal/apollo_html_reports_v1/apollo_html_reports.jl
a11_html_reports=/export/fs01/vmanoha1/fearless_steps/apollo_journal/a11_html_reports_v1/a11_html_reports.jl

dir=data/local/nasa
stage=0

. utils/parse_options.sh

mkdir -p $dir

if [ $stage -le 0 ]; then
  for m in $spacelog_missions; do
    for x in TEC PAO ATG en CM; do 
      f=$spacelog_missions_dir/$m/transcripts/$x
      if [ -f $f ]; then
        local/nasa/parse_spacelog_transcripts.py $f /dev/stdout
      fi
    done > $dir/spacelog_${m}_transcripts.txt
  done
fi

if [ $stage -le 1 ]; then
  for m in $spacelog_missions; do
    cat $dir/spacelog_${m}_transcripts.txt | local/dict/normalize_abbrv.py $dict/lexicon.txt
  done > $dir/spacelog_transcripts.txt
fi

if [ $stage -le 2 ]; then
  local/nasa/parse_nasa_reports.py $apollo_html_reports /dev/stdout | \
    local/dict/normalize_abbrv.py $dict/lexicon.txt > $dir/apollo_html_reports.txt
  local/nasa/parse_nasa_reports.py $a11_html_reports /dev/stdout | \
    local/dict/normalize_abbrv.py $dict/lexicon.txt > $dir/a11_html_reports.txt
fi

if [ $stage -le 3 ]; then
  for m in $alsj_missions; do
    local/nasa/parse_apollo_transcripts.py \
      $alsj_missions_dir/${m}_transcripts.jl /dev/stdout
  done | local/dict/normalize_abbrv.py $dict/lexicon.txt > $dir/alsj_transcripts.txt
fi

if [ $stage -le 4 ]; then
  for m in $afj_missions; do
    local/apollo_flight_journal/parse_afj_transcripts.py \
      $afj_missions_dir/apollo${m}_transcripts.jl /dev/stdout
  done | local/dict/normalize_abbrv.py $dict/lexicon.txt > $dir/afj_transcripts.txt
fi

if [ $stage -le 5 ]; then
  for m in a11; do
    local/nasa/parse_apollo_transcripts.py \
      $alsj_missions_dir/${m}_transcripts.jl /dev/stdout
  done | local/dict/normalize_abbrv.py $dict/lexicon.txt > $dir/alsj_a11_transcripts.txt
fi

if [ $stage -le 6 ]; then
  for m in 11; do
    local/apollo_flight_journal/parse_afj_transcripts.py \
      $afj_missions_dir/apollo${m}_transcripts.jl /dev/stdout
  done | local/dict/normalize_abbrv.py $dict/lexicon.txt > $dir/afj_a11_transcripts.txt
fi
