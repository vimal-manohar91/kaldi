#!/bin/bash
# Copyright 2015-2016  Sarah Flora Juan
# Copyright 2016  Johns Hopkins University (Author: Yenda Trmal)
# Apache 2.0

# config
order=4
unk="<UNK>"
# end config
. ./utils/parse_options.sh

set -e -o pipefail

# To create G.fst from ARPA language model
. ./path.sh || die "path.sh expected";

text=$1
input1=$2
input2=$3
output=$4

mkdir -p $output
for w in 1.0 0.95 0.9 0.85  0.8 0.7 0.6 0.5; do
  ngram -order $order -lm $input1  -mix-lm $input2 -unk -vocab $(dirname $input1)/vocab -map-unk "$unk" \
          -lambda $w -write-lm $output/lm.${order}gram.${w}.gz
    echo -n "$output/lm.${order}gram.${w}.gz "
    ngram -order $order  -unk -map-unk "$unk" -lm $output/lm.${order}gram.${w}.gz -ppl $text | paste -s -
done | sort  -k15,15g  > $output/perplexities.${order}gram.txt

cat $output/perplexities.${order}gram.txt | head -n 1

outlm=best_${order}gram.gz
lmfilename=$(cat $output/perplexities.${order}gram.txt | head -n 1 | cut -f 1 -d ' ')
echo "$outlm -> $lmfilename"
(cd $output; rm -f $outlm; ln -sf $(basename $lmfilename) $outlm )
exit 0;
