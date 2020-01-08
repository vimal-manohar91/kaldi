#!/bin/bash
#

if [ -f path.sh ]; then . ./path.sh; fi

arpa_lm=data/local/fisher_lm/3gram-mincount/lm_unpruned.gz
large_lm=data/local/fisher_lm/4gram-mincount/lm_unpruned.gz
lang=data/lang_fisher
dir=data/lang_fisher_test

. utils/parse_options.sh

mkdir -p $dir

[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;

cp -rT $lang $dir

gunzip -c "$arpa_lm" | \
   arpa2fst --disambig-symbol=#0 \
            --read-symbol-table=$dir/words.txt - $dir/G.fst

echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic $dir/G.fst

## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
fstprint   --isymbols=$lang/phones.txt --osymbols=$lang/words.txt $lang/L.fst  | head

echo Performing further checks

# Checking that G.fst is determinizable.
fstdeterminize $dir/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize $dir/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
# Note: we do this with fstdeterminizestar not fstdeterminize, as
# fstdeterminize was taking forever (presumbaly relates to a bug
# in this version of OpenFst that makes determinization slow for
# some case).
fsttablecompose $dir/L_disambig.fst $dir/G.fst | \
   fstdeterminizestar >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose $lang/L_disambig.fst $dir/G.fst | \
   fstisstochastic || echo "[log:] LG is not stochastic"

if [ ! -z "$large_lm" ]; then
utils/build_const_arpa_lm.sh \
    $large_lm $lang ${dir}_fg
fi

echo "$0 succeeded"
