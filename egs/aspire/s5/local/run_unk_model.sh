#!/bin/bash

# Copyright 2017  Vimal Manohar

dict_dir=data/local/dict
lang_suffix=
dir=exp/unk_lang_model

set -e -o pipefail -u

. utils/parse_options.sh

utils/lang/make_unk_lm.sh $dict_dir $dir || exit 1

utils/prepare_lang.sh \
  --unk-fst $dir/unk_fst.txt \
  $dict_dir "<unk>" data/local/lang${lang_suffix} data/lang${lang_suffix}_unk

# note: it's important that the LM we built in data/lang/G.fst was created using
# pocolm with the option --limit-unk-history=true (see ted_train_lm.sh).  This
# keeps the graph compact after adding the unk model (we only have to add one
# copy of it).

# for lang_dir in $lang_dirs; do
#   rm -r ${lang_dir}_unk 2>/dev/null || true
#   mkdir -p ${lang_dir}_unk
#   cp -r data/lang_unk ${lang_dir}_unk
#   if [ -f ${lang_dir}/G.fst ]; then cp ${lang_dir}/G.fst ${lang_dir}_unk/G.fst; fi
#   if [ -f ${lang_dir}/G.carpa ]; then cp ${lang_dir}/G.carpa ${lang_dir}_unk/G.carpa; fi
# done

exit 0

## Caution: if you use this unk-model stuff, be sure that the scoring script
## does not use lattice-align-words-lexicon, because it's not compatible with
## the unk-model.  Instead you should use lattice-align-words (of course, this
## only works if you have position-dependent phones).
