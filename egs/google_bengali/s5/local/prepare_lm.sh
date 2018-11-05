#!/bin/bash
# Copyright (c) 2018, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
stage=0
unk="<UNK>"
# End configuration section
. ./utils/parse_options.sh

[ ! -f ./path.sh ] && echo >&2 "path.sh expected to exist" && exit 1;
. ./path.sh

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

if [ $stage -le 0 ]; then
	local/train_lms_srilm.sh --oov-symbol "$unk" \
		--train-text data/train/text \
		--dev-text data/dev/text data/ data/srilm/
fi

if [ $stage -le 1 ]; then
	utils/lang/make_unk_lm.sh data/local/dict exp/make_unk
fi

if [ $stage -le 2 ]; then
	utils/prepare_lang.sh \
		--unk-fst exp/make_unk/unk_fst.txt --phone-symbol-table data/lang/phones.txt \
		data/local/dict "$unk" data/local/lang_test data/lang_test

	utils/format_lm.sh \
		data/lang_test data/srilm/best_3gram.gz data/local/dict/lexicon.txt data/lang_test
fi

if [ $stage -le 3 ]; then
	local/train_lms_srilm.sh --oov-symbol "$unk" \
		--train-text data/local/extra_text \
		--dev-text data/dev/text data/ data/srilm_extra
fi

loc=`which ngram-count` || true;
if [ -z $loc ]; then
  if [ ! -z "$SRILM" ]; then
    if uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
      sdir=$SRILM/bin/i686-m64
    else
      sdir=$SRILM/bin/i686
    fi
  else
    if uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
      sdir=`pwd`/../../../tools/srilm/bin/i686-m64
    else
      sdir=`pwd`/../../../tools/srilm/bin/i686
    fi
  fi
  if [ -f $sdir/ngram-count ]; then
    echo Using SRILM tools from $sdir
    export PATH=$PATH:$sdir
  else
    echo You appear to not have SRILM tools installed, either on your path,
    echo or installed in $sdir.  See tools/install_srilm.sh for installation
    echo instructions.
    exit 1
  fi
fi

if [ $stage -le 4 ]; then
  mkdir -p data/srilm_interp
  for w in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.01; do
      ngram -unk -renorm -map-unk "$unk"  \
            -lm data/srilm/best_4gram.gz  \
            -mix-lm data/srilm_extra/best_4gram.gz \
            -lambda $w -order 4 -write-lm data/srilm_interp/lm.${w}.gz
      echo -n "data/srilm_interp/lm.${w}.gz "
      ngram -unk -map-unk "$unk" \
        -lm data/srilm_interp/lm.${w}.gz -order 4 -ppl data/srilm/dev.txt | paste -s -
  done | sort  -k15,15g  > data/srilm_interp/perplexities.txt
fi

if [ $stage -le 5 ]; then
  lm=$(cat data/srilm_interp/perplexities.txt | head -n1 | awk '{print $1}')
  # for decoding using bigger LM let's find which interpolated gave the most improvement
  [ -d data/lang_test_fg ] && rm -rf data/lang_test_fg
  cp -R data/lang_test data/lang_test_fg
	utils/format_lm.sh \
		data/lang_test_fg $lm data/local/dict/lexicon.txt data/lang_test_fg
	utils/build_const_arpa_lm.sh $lm data/lang_test_fg data/lang_test_fg
fi
