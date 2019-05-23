#!/bin/bash

dict=data/local/dict_afj
spacelog_missions_dir=/export/fs01/vmanoha1/fearless_steps/Spacelog/missions
missions="g3 g4 g6 g7 g8 ma6 ma7 ma8 mr3 mr4"
dir=data/local/spacelog
stage=0

. utils/parse_options.sh

mkdir -p $dir

if [ $stage -le 0 ]; then
  for m in $missions; do
    for x in TEC PAO ATG en CM; do 
      f=$spacelog_missions_dir/$m/transcripts/$x
      if [ -f $f ]; then
        local/spacelog/parse_spacelog_transcripts.py $f /dev/stdout
      fi
    done > $dir/${m}_transcripts.txt
  done
fi

if [ $stage -le 1 ]; then
  for m in $missions; do
    cat $dir/${m}_transcripts.txt | local/dict/normalize_abbrv.py $dict/lexicon.txt
  done > $dir/spacelog_transcripts.txt
fi

