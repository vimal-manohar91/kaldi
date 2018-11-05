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
if [ -f data/train/text.raw ] ; then
  cat data/train/text.raw
else
  cat data/train/text
fi | cat - data/local/extra_text | \
  sed 's/[ \t]/\t/g' | cut -f 2- | sed 's/[:., \t]/\n/g' |\
  sed -e 's/^-$//g' -e 's/^&$//g' -e 's/^?$//g' | \
  grep -v '[0-9]' | sort -u | sed '/^ *$/d' > data/local/words.txt

mkdir -p data/local/dict
[ -f data/local/dict/lexiconp.txt ] && rm data/local/dict/lexiconp.txt
./local/grapheme_lexicon.py < data/local/words.txt | \
  cat <(echo -e "<UNK> <unk>\n<SIL> <sil>\n") - | \
  sort -u | sed '/^ *$/d' > data/local/dict/lexicon.txt

local/grapheme_lexicon_stats.py < data/local/dict/lexicon.txt \
  > data/local/dict/lexicon_stats.txt

[ ! -f data/train/text.raw ] && cp data/train/text data/train/text.raw
uconv -f utf-8 -t utf-8 -x Any-NFKC data/train/text.raw |\
  sed -e 's/[:,.?!&]/ /g'   | \
  ./local/map_unks.pl -f 2- data/local/dict/lexicon.txt > data/train/text

uconv -f utf-8 -t utf-8 -x Any-NFKC data/local/extra_text |\
  sed -e 's/[:,.?!&]/ /g'   | \
  ./local/map_unks.pl -f 2- data/local/dict/lexicon.txt \
  > data/local/extra_text.cleaned

sed 's/[ \t]/\t/' data/local/dict/lexicon.txt | cut -f 2 | \
  sed 's/[ \t]/\n/g' | sed '/^ *$/d' | sort -u | grep -v '<.*>' |\
  cat - <(echo '<unk>') | sort > data/local/dict/nonsilence_phones.txt

sed 's/[ \t]/\t/' data/local/dict/lexicon.txt | cut -f 2 | \
  sed 's/[ \t]/\n/g'  | sed '/^ *$/d' | sort -u | grep  '<.*>' |\
  grep -v '<unk>' > data/local/dict/silence_phones.txt

echo "<sil>" > data/local/dict/optional_silence.txt
echo "<UNK>" > data/local/dict/oov.txt

utils/validate_dict_dir.pl data/local/dict

