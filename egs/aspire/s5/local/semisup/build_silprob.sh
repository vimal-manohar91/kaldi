#!/bin/bash

set -e

. ./cmd.sh
. ./path.sh

steps/get_prons.sh --cmd "$train_cmd" data/train_300k data/lang exp/semisup300k/tri5b

utils/dict_dir_add_pronprobs.sh --max-normalize true \
  data/local/dict exp/semisup300k/tri5b/pron_counts_nowb.txt \
  exp/semisup300k/tri5b/sil_counts_nowb.txt \
  exp/semisup300k/tri5b/pron_bigram_counts_nowb.txt data/local/dict_300k_pp

utils/prepare_lang.sh data/local/dict_300k_pp "<unk>" data/local/lang_300k_pp data/lang_300k_pp
