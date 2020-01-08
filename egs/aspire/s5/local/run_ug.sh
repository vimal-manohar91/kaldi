mkdir -p data/lang_pp_ug_test

oov=`cat data/lang_pp/oov.int` || exit 1;
cp -rT data/lang_pp data/lang_pp_ug_test

cat data/train/text | utils/sym2int.pl --map-oov $oov -f 2- data/lang_pp/words.txt | \
  awk '{for(n=2;n<=NF;n++){ printf("%s ", $n); } printf("\n"); }' | \
  utils/make_unigram_grammar.pl | fstcompile | fstarcsort --sort_type=ilabel > data/lang_pp_ug_test/G.fst \
   || exit 1;

