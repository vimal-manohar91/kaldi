#!/bin/bash

# Copyright 2017  Intellisist, Inc. (Author: Navneeth K)
#           2017  Xiaohui Zhang
# Apache License 2.0

# This script trains a g2p model using Phonetisaurus and SRILM.
set -e -o pipefail
stage=0
silence_phones=
srilm_opts="-order 7 -kn-modify-counts-at-end -gt1min 0 -gt2min 0
    -gt3min 0 -gt4min 0 -gt5min 0 -gt6min 0 -gt7min 0 -ukndiscount"

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. utils/parse_options.sh || exit 1;


if [ $# -ne 2 ] && [ $# -ne 3 ] ; then
  echo "Usage: $0 <dictdir> [lexicon] <outdir>"
  exit 1;
fi

if [ $# -eq 2 ] ; then
  lexicondir=$1
  lexicon=$lexicondir/lexicon.txt
  outdir=$2
else
  lexicondir=$1
  lexicon=$2
  outdir=$3
fi

[ ! -f $lexicon ] && echo "Cannot find $lexicon" && exit

isuconv=`which uconv`
if [ -z $isuconv ]; then
  echo "uconv was not found. You must install the icu4c package."
  exit 1;
fi
isphon=`which phonetisaurus-align`
if [ -z $isphon ]; then
  echo "phonetisaurus-align  was not found. You must install the phonetisaurus-g2p package."
  exit 1;
fi

mkdir -p $outdir


# For input lexicon, remove pronunciations containing non-utf-8-encodable characters,
# and optionally remove words that are mapped to a single silence phone from the lexicon.
if [ $stage -le 0 ]; then
  if [ ! -z "$silence_phones" ]; then
    awk 'NR==FNR{a[$1] = 1; next} {s=$2;for(i=3;i<=NF;i++) s=s" "$i; if(!(s in a)) print $1" "s}' \
      $silence_phones $lexicon | \
      awk '{printf("%s\t",$1); for (i=2;i<NF;i++){printf("%s ",$i);} printf("%s\n",$NF);}' | \
      uconv -f utf-8  -t utf-8 -x Any-NFC - | awk 'NF > 0'> $outdir/lexicon_tab_separated.txt
  else
    awk '{printf("%s\t",$1); for (i=2;i<NF;i++){printf("%s ",$i);} printf("%s\n",$NF);}' $lexicon | \
      uconv -f utf-8  -t utf-8 -x Any-NFC - | awk 'NF > 0'> $outdir/lexicon_tab_separated.txt
  fi
fi
set -x
if [ $stage -le 1 ]; then
  cat $outdir/lexicon_tab_separated.txt | perl -mutf8 -CS -ne '
    chomp;
    @F = split " ", $_;
    $F[0] =~ s/\.$//;
    $word = shift @F;
    $pron = join("|", @F);
    $pron =~ s/[0-9]//g;
    print "$word}$pron .}_\n";
  ' > $outdir/aligned_lexicon.corpus
fi

if [ $stage -le 2 ]; then
  # Convert aligned lexicon to arpa using make_kn_lm.py, a re-implementation of srilm's ngram-count functionality.
  ./utils/lang/make_kn_lm.py -ngram-order 2 \
    -text ${outdir}/aligned_lexicon.corpus -lm ${outdir}/aligned_lexicon.arpa
fi

if [ $stage -le 3 ]; then
  # Convert the arpa file to FST.
  phonetisaurus-arpa2wfst --lm=${outdir}/aligned_lexicon.arpa --ofile=${outdir}/model.fst
fi
