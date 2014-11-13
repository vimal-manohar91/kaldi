#!/bin/bash

# Fisher unsupervised training recipe

# It's best to run the commands in this one by one.

. cmd.sh
. path.sh

mfccdir=`pwd`/mfcc
nj=50
unsup_nj=50

set -e

false && {
utils/subset_data_dir.sh --spk-list <(utils/filter_scp.pl --exclude data/train_100k/spk2utt data/train/spk2utt) data/train data/unsup_100k
utils/subset_data_dir.sh --speakers data/unsup_100k 250000 data/unsup_100k_250k
}

local/fisher_train_lms.sh --text data/train_100k/text --dir data/local/lm_100k
local/fisher_create_test_lang.sh --lmdir data/local/lm_100k --lang data/lang_100k_test

false && {
steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k data/lang exp/tri2 exp/tri2_ali || exit 1;

# Train tri3a, which is LDA+MLLT, on 100k data.
steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   5000 40000 data/train_100k data/lang exp/tri2_ali exp/tri3a || exit 1;
(
  utils/mkgraph.sh data/lang_100k_test exp/tri3a exp/tri3a/graph_100k || exit 1;
  steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
    --num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G" \
   exp/tri3a/graph_100k data/dev exp/tri3a/decode_100k_dev || exit 1;
)&


# Next we'll use fMLLR and train with SAT (i.e. on 
# fMLLR features)

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k data/lang exp/tri3a exp/tri3a_ali || exit 1;

steps/train_sat.sh  --cmd "$train_cmd" \
  5000 100000 data/train_100k data/lang exp/tri3a_ali  exp/tri4a || exit 1;

(
  utils/mkgraph.sh data/lang_100k_test exp/tri4a exp/tri4a/graph_100k
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
    --num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G" \
   exp/tri4a/graph_100k data/dev exp/tri4a/decode_100k_dev
)&
}

(
  utils/mkgraph.sh data/lang_100k_test exp/tri5a exp/tri5a/graph_100k
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
    --num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G" \
    exp/tri5a/graph_100k data/dev exp/tri5a/decode_100k_dev
)&

# Run the steps below to prepare for unsupervised training

steps/decode_fmllr.sh --nj $unsup_nj --cmd "$decode_cmd" \
  --config conf/decode.config \
  --num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G" \
  exp/tri4a/graph_100k data/unsup_100k exp/tri4a/decode_100k_unsup_100k || exit 1

split_data.sh data/unsup_100k_250k 32 || exit 1

mkdir -p exp/tri4a/decode_100k_unsup_100k_250k
lattices=$(eval echo exp/tri4a/decode_100k_unsup_100k/lat.{`seq -s',' $unsup_nj`}.gz)

nj=32

for n in `seq $nj`; do
  $decode_cmd JOB=1:$unsup_nj exp/tri4a/decode_100k_unsup_100k_250k/log/filter_lattices.$n.JOB.log \
    lattice-filter-copy --force=true data/unsup_100k_250k/split32/$n/segments \
    "ark,s,cs:gunzip -c exp/tri4a/decode_100k_unsup_100k/lat.JOB.gz |" \
    "ark:| gzip -c > exp/tri4a/decode_100k_unsup_100k_250k/lat.$n.JOB.gz" || exit 1
  cat $(eval echo exp/tri4a/decode_100k_unsup_100k_250k/lat.$n.{`seq -s ',' $unsup_nj`}.gz) > exp/tri4a/decode_100k_unsup_100k_250k/lat.$n.gz
done

trans=$(eval echo exp/tri4a/decode_100k_unsup_100k/trans.{`seq -s',' $unsup_nj`})
$train_cmd JOB=1:32 exp/tri4a/decode_100k_unsup_100k_250k/log/filter_trans.JOB.gz \
  filter-copy-matrix data/unsup_100k_250k/split32/JOB/spk2utt \
  "ark,s,cs:cat $trans |" \
  ark:exp/tri4a/decode_100k_unsup_100k_250k/trans.JOB || exit 1

echo 32 > exp/tri4a/decode_100k_unsup_100k_250k/num_jobs

local/run_nce.sh --num-iters 12 --tau 800
