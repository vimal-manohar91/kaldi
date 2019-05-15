#!/bin/bash

# Copyright 2019  Vimal Manohar
# Apache 2.0

stage=0
orig_dict_dir=data/local/dict
train_text=data/train/text
extra_text=data/apollo11_whole_legal/text
dir=data/local/dict_apollo11

. path.sh
. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  exit 1
fi

orig_dict=$orig_dict_dir/lexicon.txt
if [ ! -f $orig_dict ]; then
  echo "$0: Could not read $orig_dict"
  exit 1
fi

mkdir -p $dir

if [ $stage -le 0 ]; then
  if [ ! -d $dir/cmudict ]; then
    svn co https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict  $dir/cmudict
  fi
fi

if [ $stage -le 1 ]; then
  # silence phones, one per line.
  for w in sil laughter noise oov; do echo $w; done > $dir/silence_phones.txt
  echo sil > $dir/optional_silence.txt

  # For this setup we're discarding stress.
  cat $dir/cmudict/cmudict.0.7a.symbols | sed 's/[0-9]//g' | \
   tr '[A-Z]' '[a-z]' | perl -ane 's:\r::; print;' | sort | uniq > $dir/nonsilence_phones.txt

  # An extra question will be added by including the silence phones in one class.
  cat $dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;

  grep -v ';;;' $dir/cmudict/cmudict.0.7a |  tr '[A-Z]' '[a-z]' | \
   perl -ane 'if(!m:^;;;:){ s:(\S+)\(\d+\) :$1 :; s:  : :; print; }' | \
   perl -ane '@A = split(" ", $_); for ($n = 1; $n<@A;$n++) { $A[$n] =~ s/[0-9]//g; } print join(" ", @A) . "\n";' | \
   sort | uniq > $dir/lexicon1_raw_nosil.txt || exit 1;

  # Add prons for laughter, noise, oov
  for w in `grep -v sil $dir/silence_phones.txt`; do
    echo "<$w> $w"
  done | cat - $dir/lexicon1_raw_nosil.txt  > $dir/lexicon2_raw.txt || exit 1;

  #we let only the spelled letter in
  (set +o pipefail; grep   '^[a-z]\.[\t ]' $dir/lexicon1_raw_nosil.txt | grep  -v -F ';;;' > $dir/dict_abbrv) || true
fi

if [ $stage -le 2 ]; then
  steps/dict/train_g2p_phonetisaurus.sh \
    --silence-phones "$dir/silence_phones.txt" \
    $dir/lexicon1_raw_nosil.txt exp/g2p || exit 1
fi

if [ $stage -le 3 ]; then
  local/g2p/train_g2p_abbrv.sh \
    --silence-phones "$dir/silence_phones.txt" \
    $dir $dir/dict_abbrv exp/g2p_abbrv || exit 1
fi

if [ $stage -le 4 ]; then
  (
  cat $train_text | cut -d ' ' -f 2- | \
    local/count_oovs.pl $dir/lexicon1_raw_nosil.txt | \
    awk '{if (NF > 3 ) {for(i=4; i<NF; i++) printf "%s ",$i; print $NF;}}' | \
    perl -ape 's/\s/\n/g;' | \
    sort | uniq -c | \
    awk '{if ($1 > 1) print $2}' 

  cat $extra_text | cut -d ' ' -f 2- | \
    local/count_oovs.pl $dir/lexicon1_raw_nosil.txt | \
    awk '{if (NF > 3 ) {for(i=4; i<NF; i++) printf "%s ",$i; print $NF;}}' | \
    perl -ape 's/\s/\n/g;' | \
    sort | uniq -c | \
    awk '{if ($1 > 2) print $2}' 
  ) | sort -u | \
    grep -v "<unk>" | grep -v "<laughter>" > $dir/words_oov.txt

  perl -ne 'if (m/[a-z]\./) {print $_;} else {print STDERR $_;}' $dir/words_oov.txt \
    > $dir/words_oov_abbrv.txt 2> $dir/words_oov_normal.txt
fi

if [ $stage -le 5 ]; then
  steps/dict/apply_g2p_phonetisaurus.sh --nbest 1 \
    $dir/words_oov_abbrv.txt exp/g2p_abbrv exp/g2p_abbrv/oov_lex

  steps/dict/apply_g2p_phonetisaurus.sh --nbest 1 \
    $dir/words_oov_normal.txt exp/g2p exp/g2p/oov_lex

  rm -f $dir/lexicon.txt
  (
    cat $orig_dict_dir/lexiconp.txt
    cat exp/g2p_abbrv/oov_lex/lexicon.lex 
    cat exp/g2p/oov_lex/lexicon.lex 
  ) | sort -k1,1 > $dir/lexiconp.txt
fi
