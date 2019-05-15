#!/bin/bash

# Copyright 2016  Allen Guo
#           2017  Xiaohui Zhang
# Apache License 2.0

# This script applies a trained Phonetisarus G2P model to
# synthesize pronunciations for missing words (i.e., words in
# transcripts but not the lexicon), and output the expanded lexicon.

var_counts=1
set +o pipefail
set -x

. ./path.sh || exit 1
. parse_options.sh || exit 1;

if [ $# -ne "5" ]; then
  echo "Usage: $0 <input-text> <g2p-model> <g2p-tmp-dir> <current-lexicon> <output-lexicon>"
  exit 1
fi

input=$1
model=$2
workdir=$3
lexicon=$4
outlexicon=$5

mkdir -p $workdir


echo 'Synthesizing pronunciations for missing words...'
phonetisaurus-apply --nbest $var_counts --model $model/model.fst --thresh 5 --accumulate --word_list $input > $workdir/missing_g2p_${var_counts}.txt

echo "Extending $lexicon by new pronunciations, output to $outlexicon"
cat "$lexicon" $workdir/missing_g2p_${var_counts}.txt | sort | uniq > $outlexicon
