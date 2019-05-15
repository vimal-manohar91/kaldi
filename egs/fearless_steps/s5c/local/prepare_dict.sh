#!/bin/bash
# Copyright (c) 2019, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e
set -o pipefail
set -o nounset                              # Treat unset variables as an error


train_text=data/train/text
extra_text=data/apollo11_whole_legal/text

dir=data/local/dict

mkdir -p $dir

[ -f $dir/cmudict ] ||
  (wget -q -O /dev/stdout http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b | \
   uconv -f iso-8859-1 -t utf-8 | tr '[a-z]' '[A-Z]' > $dir/cmudict) || true

cat $train_text | cut -d ' ' -f 2- | tr ' ' '\n' | sort | uniq -c | \
  perl -ane 'if ($F[0] > 1) { print "$F[1]\n"; }' > $dir/words_train.txt

cat $extra_text | cut -d ' ' -f 2- | tr ' ' '\n' | sort | uniq -c | \
  perl -ane 'if ($F[0] > 1) { print "$F[1]\n"; }' > $dir/words_extra.txt

(
  echo '<sil>'
  cat $dir/words_train.txt
  cat $dir/words_extra.txt
) | sort -u | awk '{print $0" "NR}' > $dir/words.txt

echo '<sil>' > $dir/oov.txt

awk '{print $1}' $dir/words.txt  | \
	python3 local/cmu_lexicon.py $dir/cmudict >$dir/lexicon_iv.txt 2>$dir/words_oov.txt

#we let only the spelled letter in
(set +o pipefail; grep   '^[A-Z]\.[\t ]' $dir/cmudict | grep  -v -F ';;;' > $dir/cmudict_abbrv) || true

./local/g2p/train_g2p.sh \
  $dir $dir/cmudict data/local/g2p

./local/g2p/train_g2p_abbrv.sh --srilm-opts "-order 2 -wbdiscount"  \
  $dir/ $dir/cmudict_abbrv data/local/g2p_abbrv

./local/g2p/apply_g2p.sh  \
   $dir/words_oov.txt  data/local/g2p_abbrv $dir/g2p_abbrv \
   $dir/lexicon_iv.txt $dir/lexicon_iv_oov_abbrv.txt

./local/g2p/apply_g2p.sh  \
   $dir/words_oov.txt  data/local/g2p $dir/g2p \
   $dir/lexicon_iv_oov_abbrv.txt $dir/lexicon.txt

sed 's/^[^ \t][^ \t]*[ \t]//;' $dir/lexicon.txt | \
  sed -e 's/ /\n/g' | sort -u | sed '/^[ \t]*$/d'  > $dir/phones.txt

grep '<' $dir/phones.txt > $dir/silence_phones.txt
grep -v '<' $dir/phones.txt > $dir/nonsilence_phones.txt
echo '<sil>' > $dir/optional_silence.txt

utils/validate_dict_dir.pl $dir
