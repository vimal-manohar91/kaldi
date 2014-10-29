#!/bin/bash

# Fisher + Switchboard combined recipe, adapted from respective Fisher and Switchboard
# recipes by Peng Qi (pengqi@cs.stanford.edu).
# (Aug 2014)

# It's best to run the commands in this one by one.

. cmd.sh
. path.sh
mfccdir=`pwd`/mfcc
nj=50
unsup_nj=50

set -e

false && {
steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k data/lang exp/tri2 exp/tri2_ali || exit 1;

# Train tri3a, which is LDA+MLLT, on 100k data.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   5000 40000 data/train_100k data/lang exp/tri2_ali exp/tri3a || exit 1;
(
  utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
  steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri3a/graph data/dev exp/tri3a/decode_dev || exit 1;
)&


# Next we'll use fMLLR and train with SAT (i.e. on 
# fMLLR features)

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k data/lang exp/tri3a exp/tri3a_ali || exit 1;

steps/train_sat.sh  --cmd "$train_cmd" \
  5000 100000 data/train_100k data/lang exp/tri3a_ali  exp/tri4a || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri4a/graph data/dev exp/tri4a/decode_dev
)&
}

steps/decode_fmllr.sh --nj $unsup_nj --cmd "$decode_cmd" \
  --config conf/decode.config \
  --num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G" \
  exp/tri4a/graph data/unsup_100k exp/tri4a/decode_unsup_100k || exit 1

# The step below won't run by default; it demonstrates a data-cleaning method.
# It doesn't seem to help in this setup; maybe the data was clean enough already.
# local/run_data_cleaning.sh

# local/run_for_spkid.sh

# local/run_nnet2.sh

# local/online/run_nnet2.sh

