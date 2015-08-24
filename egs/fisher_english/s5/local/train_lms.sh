#!/bin/bash


# To be run from one directory above this script.

. path.sh

text=data/train_all/text
lexicon=data/local/dict/lexicon.txt 
dir=data/local/lm
heldout_sent=10000 
lm_order=3

. parse_options.sh

for f in "$text" "$lexicon"; do
  [ ! -f $x ] && echo "$0: No such file $f" && exit 1;
done

# This script takes no arguments.  It assumes you have already run
# fisher_data_prep.sh and fisher_prepare_dict.sh
# It takes as input the files
#data/train_all/text
#data/local/dict/lexicon.txt

mkdir -p $dir

cat $text | awk '{for(n=2;n<=NF;n++){ printf $n; if(n<NF) printf " "; else print ""; }}' | \
  head -$heldout_sent > $dir/heldout
cat $text | awk '{for(n=2;n<=NF;n++){ printf $n; if(n<NF) printf " "; else print ""; }}' | \
  tail -n +$heldout_sent > $dir/train

cat $lexicon | awk '{print $1}' | cat - <(echo "<s>"; echo "</s>"; echo "<unk>" ) | sort -u > $dir/wordlist

ngram-count -text $dir/train -order $lm_order -limit-vocab -vocab $dir/wordlist -unk \
  -map-unk "<unk>" -kndiscount -interpolate -lm $dir/srilm.o${lm_order}g.kn.gz
ngram -lm $dir/srilm.o${lm_order}g.kn.gz -ppl $dir/heldout 

# data/local/lm/srilm/srilm.o3g.kn.gz: line 71: warning: non-zero probability for <unk> in closed-vocabulary LM
# file data/local/lm/srilm/heldout: 10000 sentences, 78998 words, 0 OOVs
# 0 zeroprobs, logprob= -165170 ppl= 71.7609 ppl1= 123.258

# data/local/lm/srilm/srilm.o3g.kn.gz: line 71: warning: non-zero probability for <unk> in closed-vocabulary LM
# file data/local/lm/srilm/heldout: 10000 sentences, 78998 words, 0 OOVs
# 0 zeroprobs, logprob= -164990 ppl= 71.4278 ppl1= 122.614
