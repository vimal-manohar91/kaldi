false && {
local/fisher_train_lms_pocolm.sh --text data/train_300k_dev/text --lexicon data/local/dict/lexicon.txt --dir data/local/pocolm_300k --num-ngrams-large 250000

local/fisher_create_test_lang.sh --arpa-lm data/local/pocolm_300k/data/arpa/4gram_big.arpa.gz --lang data/lang_300k_pp --dir data/lang_300k_pp_test

local/semisup/build_silprob.sh 
}

mkdir -p data/lang_300k_pp_ug_test

oov=`cat data/lang_300k_pp/oov.int` || exit 1;
cp -rT data/lang_300k_pp data/lang_300k_pp_ug_test

cat data/train_300k_dev/text | utils/sym2int.pl --map-oov $oov -f 2- data/lang_300k_pp/words.txt | \
  awk '{for(n=2;n<=NF;n++){ printf("%s ", $n); } printf("\n"); }' | \
  utils/make_unigram_grammar.pl | fstcompile | fstarcsort --sort_type=ilabel > data/lang_300k_pp_ug_test/G.fst \
   || exit 1;

