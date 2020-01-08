utils/lang/make_unk_lm.sh data/local/dictp/tri5_ali exp/unk_lang_model

utils/prepare_lang.sh \
  --unk-fst exp/unk_lang_model/unk_fst.txt \
  data/local/dictp/tri5_ali "<unk>" data/local/langp_unk data/langp_unk

gunzip -c data/srilm/lm.gz | utils/lang/limit_arpa_unk_history.py "<unk>" | \
  gzip -c > data/srilm/lm_unklimit.gz

local/arpa2G.sh data/srilm/lm_unklimit.gz data/langp_unk data/langp_unk_test
