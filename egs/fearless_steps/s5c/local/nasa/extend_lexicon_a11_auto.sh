#!/bin/bash

# Copyright 2019  Vimal Manohar
# Apache 2.0

stage=0
orig_dict_dir=data/local/dict
train_text=data/train/text
a11_text=data/apollo11_whole_legal/text
extra_transcripts="data/local/nasa_v1/afj_transcripts.txt data/local/nasa_v1/alsj_transcripts.txt data/local/nasa_v1/spacelog_transcripts.txt data/local/nasa_v1/apollo_html_reports.txt data/local/nasa_v1/a11_html_reports.txt"
g2p_dir=
dir=data/local/dict_nasa_v1

if [ -f ./path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh

set -o pipefail

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
  (set +o pipefail; grep '^[a-z]\.[\t ]' $dir/lexicon1_raw_nosil.txt | grep  -v -F ';;;' > $dir/dict_abbrv) || true
fi

if [ -z "$g2p_dir" ]; then
  g2p_dir=exp/g2p
  if [ $stage -le 2 ]; then
    steps/dict/train_g2p_phonetisaurus.sh \
      --silence-phones "$dir/silence_phones.txt" \
      $dir/lexicon1_raw_nosil.txt $g2p_dir || exit 1
  fi
fi

if [ $stage -le 4 ]; then
  (
    #local/count_oovs.pl $dir/lexicon1_raw_nosil.txt | \
    #local/count_oovs.pl $orig_dict_dir/lexicon.txt | \
  cat $train_text $a11_text | cut -d ' ' -f 2- | \
    local/count_oovs.pl $dir/lexicon1_raw_nosil.txt | \
    awk '{if (NF > 3 ) {for(i=4; i<NF; i++) printf "%s ",$i; print $NF;}}' | \
    perl -ape 's/\s/\n/g;' | \
    sort | uniq -c | \
    awk '{if ($1 > 1) print $2}' 

  for f in $extra_transcripts; do
    cat $f | \
      local/count_oovs.pl $dir/lexicon1_raw_nosil.txt | \
      awk '{if (NF > 3 ) {for(i=4; i<NF; i++) printf "%s ",$i; print $NF;}}' | \
      perl -ape 's/\s/\n/g;' | \
      sort | uniq -c | \
      awk '{if ($1 > 1) print $2}' 
  done
  ) | sort -u | \
    grep -v "<unk>" | grep -v "<laughter>" | grep -v "<noise>" > $dir/words_oov.txt

  (
  cut -d ' ' -f 2- $train_text
  for f in $extra_transcripts; do cat $f; done
  ) | perl -e '
  my %words;
  while (<>) {
    chomp;
    my @F = split;
    foreach my $w (@F) {
      $words{$w} = 1;
    }
  }

  foreach my $w (keys %words) {
    print "$w\n";
  }' > $dir/words_train.txt

  (
  awk '{print $1}' $orig_dict_dir/lexicon.txt 
  cat $dir/words_train.txt
  ) | sort | uniq > $dir/words_all.txt
fi

if [ $stage -le 5 ]; then
  cat $dir/words_oov.txt | grep -v '[0-9]' | local/dict/get_abbrv_words.pl \
    $dir/words_oov_abbrv.txt $dir/words_oov_normal.txt
fi

if [ $stage -le 6 ]; then
  steps/dict/apply_g2p_phonetisaurus.sh --nbest 1 \
    $dir/words_oov_normal.txt exp/g2p $dir/oov_lex || exit 1
 
  (
    perl -ane 'print "$F[0] "; shift @F; shift @F; print (join(" ", @F) . "\n")' $dir/oov_lex/lexicon.lex
    cat $dir/lexicon2_raw.txt
    cat $dir/dict_abbrv
    echo "mm m"
    echo "<unk> oov"
    echo "[laughter] laughter"
    echo "[noise] noise"
  ) | sort | uniq > $dir/lexicon3.txt
fi

if [ $stage -le 7 ]; then
   local/dict/get_prons_for_abbrv.pl \
     $dir/lexicon3.txt $dir/parts_oov.txt $dir/words_oov_abbrv_not_found.txt \
     < $dir/words_oov_abbrv.txt > $dir/abbrv_lexicon.txt || exit 1
fi

if [ $stage -le 8 ]; then
  # get pronunciations for parts for which we do not have pronunciations
  steps/dict/apply_g2p_phonetisaurus.sh --nbest 1 \
    $dir/parts_oov.txt exp/g2p $dir/parts_oov_lex || exit 1

  (
    cat $dir/lexicon3.txt
    cat $dir/abbrv_lexicon.txt
    perl -ane 'print "$F[0] "; shift @F; shift @F; print (join(" ", @F) . "\n")' $dir/parts_oov_lex/lexicon.lex 
  ) | sort -k1,1 > $dir/lexicon4.txt

  local/dict/get_prons_for_abbrv.pl \
    $dir/lexicon4.txt /dev/null /dev/null \
    < $dir/words_oov_abbrv_not_found.txt > $dir/abbrv_lexicon2.txt || exit 1
  
  rm -f $dir/lexicon.txt
  rm -f $dir/lexiconp.txt
  (
    cat $dir/lexicon4.txt
    cat $dir/abbrv_lexicon2.txt
  ) | sort | uniq | local/dict/filter_words.pl $dir/words_all.txt \
    > $dir/lexicon.txt || exit 1
fi

