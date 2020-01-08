stage=1

. ./path.sh

. utils/parse_options.sh

if [ $stage -le 1 ]; then
  local/tedlium_prepare_dict.sh
fi

if [ $stage -le 2 ]; then
  utils/lang/make_unk_lm.sh data/local/tedlium_dict_nosp exp/tedlium_unk_model
fi

if [ $stage -le 3 ]; then
  #wget --continue http://kaldi-asr.org/models/5/4gram_small.arpa.gz -P data/local/tedlium_lm/data/arpa
  #wget --continue http://kaldi-asr.org/models/5/4gram_big.arpa.gz -P data/local/tedlium_lm/data/arpa

  local/ted_prepare_lm.sh
fi

if [ $stage -le 4 ]; then
  utils/prepare_lang.sh --unk-fst exp/tedlium_unk_model/unk_fst.txt \
    data/local/tedlium_dict_nosp \
    "<unk>" data/local/tedlium_lang_unk_nosp data/tedlium_lang_unk_nosp
fi

lang=data/ted_lang_unk_nosp

cp -rT data/tedlium_lang_unk_nosp $lang
rm $lang/G.fst
lang_rescore=${lang}_rescore

small_arpa_lm=data/local/ted_lm/data/arpa/4gram_small.arpa.gz
big_arpa_lm=data/local/ted_lm/data/arpa/4gram_big.arpa.gz

for f in $small_arpa_lm $big_arpa_lm $lang/words.txt; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 5 ]; then
  if [ -f $lang/G.fst ] && [ $lang/G.fst -nt $small_arpa_lm ]; then
    echo "$0: not regenerating $lang/G.fst as it already exists and "
    echo ".. is newer than the source LM."
  else
    arpa2fst --disambig-symbol=#0 --read-symbol-table=$lang/words.txt \
      "gunzip -c $small_arpa_lm| utils/lang/limit_arpa_unk_history.py '<unk>' |" $lang/G.fst
    echo  "$0: Checking how stochastic G is (the first of these numbers should be small):"
    fstisstochastic $lang/G.fst || true
    utils/validate_lang.pl --skip-determinization-check $lang
  fi

fi

if [ $stage -le 6 ]; then
  if [ -f $lang_rescore/G.carpa ] && [ $lang_rescore/G.carpa -nt $big_arpa_lm ] && \
    [ $lang_rescore/G.carpa -nt $lang/words.txt ]; then
  echo "$0: not regenerating $lang_rescore/ as it seems to already by up to date."
else
  utils/build_const_arpa_lm.sh $big_arpa_lm $lang $lang_rescore || exit 1;
fi
fi

exit 0;
