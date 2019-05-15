#!/bin/bash
# Copyright (c) 2019, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
#                     Vimal Manohar
# License: Apache 2.0

# Begin configuration section.
stage=0

# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

# Challenge data
AUDIO=/export/corpora5/Fearless_Steps/Data/Audio/Tracks/
TRANSCRIPTS=/export/corpora5/Fearless_Steps/Data/Transcripts/ASR/

# Full data
FULL_CORPUS=/export/fs01/jtrmal/media/

if [ $stage -le 0 ]; then
	local/prepare_data.sh $AUDIO $TRANSCRIPTS
  local/prepare_full_corpus.sh $FULL_CORPUS
fi

if [ $stage -le 1 ]; then
  local/nasa/prepare_transcripts.sh \
    --dict data/local/dict --dir data/local/nasa_v1
fi

if [ $stage -le 2 ]; then
  local/nasa/extend_lexicon.sh \
    --orig-dict-dir data/local/dict \
    --train-text data/train/text \
    --dir data/local/dict_nasa_v1 \
    --g2p-dir exp/g2p
  utils/validate_dict_dir.pl data/local/dict_nasa_v1
fi

if [ $stage -le 3 ]; then
  local/nasa/prepare_transcripts.sh \
    --dict data/local/dict_nasa_v1 --dir data/local/nasa_v2
fi

if [ $stage -le 3 ]; then
  local/nasa/extend_lexicon.sh \
    --orig-dict-dir data/local/dict_nasa_v1 \
    --train-text data/train/text \
    --dir data/local/dict_nasa_v2 \
    --g2p-dir exp/g2p
  utils/validate_dict_dir.pl data/local/dict_nasa_v2
fi

if [ $stage -le 4 ]; then
  utils/prepare_lang.sh data/local/dict_nasa_v2 '<unk>' data/local/lang data/lang
fi

if [ $stage -le 5 ]; then
  #local/train_lms.sh --dir data/local/pocolm
  #utils/format_lm.sh data/lang data/local/pocolm/data/arpa/4gram_small.arpa.gz \
  #  data/local/dict_apollo11/lexicon.txt data/lang_test
  rm -rf data/local/pocolm
  local/nasa/train_lms.sh \
    --dir data/local/pocolm
fi

if [ $stage -le 3 ]; then
  utils/format_lm.sh data/lang data/local/pocolm/data/arpa/4gram_small.arpa.gz \
    data/local/dict_afj/lexicon.txt data/lang_test
fi

exit 0

if [ $stage -le 4 ] ; then
	steps/make_mfcc.sh --nj 16 --cmd "$cmd" data/train exp/make_mfcc/train mfcc
  utils/fix_data_dir.sh data/train
  steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train_cmvn mfcc
  utils/fix_data_dir.sh data/train

	steps/make_mfcc.sh --nj 16 --cmd "$cmd" data/dev exp/make_mfcc/dev mfcc
  utils/fix_data_dir.sh data/dev
  steps/compute_cmvn_stats.sh data/dev exp/make_mfcc/dev_cmvn mfcc
  utils/fix_data_dir.sh data/dev
fi

if [ $stage -le 2 ] ; then
	utils/subset_data_dir.sh data/train 1000 data/train_sub1
fi

if [ $stage -le 3 ] ; then
  echo "Starting triphone training."
  steps/train_mono.sh --nj 8 --cmd "$cmd" data/train data/lang exp/mono
  echo "Monophone training done."
fi

nj=16
dev_nj=16
if [ $stage -le 4 ]; then
  ### Triphone
  echo "Starting triphone training."
  steps/align_si.sh --nj $nj --cmd "$cmd" \
      data/train data/lang exp/mono exp/mono_ali
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$cmd"  \
      3200 30000 data/train data/lang exp/mono_ali exp/tri1
  echo "Triphone training done."

  (
  echo "Decoding the dev set using triphone models."
  #utils/mkgraph.sh data/lang_test  exp/tri1 exp/tri1/graph
  steps/decode.sh --nj $dev_nj --cmd "$cmd"  \
      exp/tri1/graph  data/dev exp/tri1/decode_dev
  echo "Triphone decoding done."
  ) &
fi

if [ $stage -le 5 ]; then
  ## Triphones + delta delta
  # Training
  echo "Starting (larger) triphone training."
  steps/align_si.sh --nj $nj --cmd "$cmd" --use-graphs true \
       data/train data/lang exp/tri1 exp/tri1_ali
  steps/train_deltas.sh --cmd "$cmd"  \
      4200 40000 data/train data/lang exp/tri1_ali exp/tri2a
  echo "Triphone (large) training done."

  (
  echo "Decoding the dev set using triphone(large) models."
  utils/mkgraph.sh data/lang_test exp/tri2a exp/tri2a/graph
  steps/decode.sh --nj $dev_nj --cmd "$cmd" \
      exp/tri2a/graph data/dev exp/tri2a/decode_dev
  ) &
fi

if [ $stage -le 6 ]; then
  ### Triphone + LDA and MLLT
  # Training
  echo "Starting LDA+MLLT training."
  steps/align_si.sh --nj $nj --cmd "$cmd"  \
      data/train data/lang exp/tri2a exp/tri2a_ali

  steps/train_lda_mllt.sh --cmd "$cmd"  \
    --splice-opts "--left-context=3 --right-context=3" \
    4200 40000 data/train data/lang exp/tri2a_ali exp/tri2b
  echo "LDA+MLLT training done."

  (
  echo "Decoding the dev set using LDA+MLLT models."
  utils/mkgraph.sh data/lang_test exp/tri2b exp/tri2b/graph
  steps/decode.sh --nj $dev_nj --cmd "$cmd" \
      exp/tri2b/graph data/dev exp/tri2b/decode_dev
  ) &
fi


if [ $stage -le 7 ]; then
  ### Triphone + LDA and MLLT + SAT and FMLLR
  # Training
  echo "Starting SAT+FMLLR training."
  steps/align_si.sh --nj $nj --cmd "$cmd" \
      --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali
  steps/train_sat.sh --cmd "$cmd" 4200 40000 \
      data/train data/lang exp/tri2b_ali exp/tri3b
  echo "SAT+FMLLR training done."

  (
  echo "Decoding the dev set using SAT+FMLLR models."
  utils/mkgraph.sh data/lang_test  exp/tri3b exp/tri3b/graph
  steps/decode_fmllr.sh --nj $dev_nj --cmd "$cmd" \
      exp/tri3b/graph  data/dev exp/tri3b/decode_dev

  echo "SAT+FMLLR decoding done."
  ) &
fi

if [ $stage -le 8 ]; then
  ### Triphone + LDA and MLLT + SAT and FMLLR
  # Training
  echo "Starting SAT+FMLLR training."
  steps/align_si.sh --nj $nj --cmd "$cmd" \
      --use-graphs true data/train data/lang exp/tri3b exp/tri3b_ali
  steps/train_sat.sh --cmd "$cmd" 4500 50000 \
      data/train data/lang exp/tri3b_ali exp/tri4b
  echo "SAT+FMLLR training done."

  (
  echo "Decoding the dev set using SAT+FMLLR models."
  utils/mkgraph.sh data/lang_test  exp/tri4b exp/tri4b/graph
  steps/decode_fmllr.sh --nj $dev_nj --cmd "$cmd" \
      exp/tri4b/graph  data/dev exp/tri4b/decode_dev

  echo "SAT+FMLLR decoding done."
  ) &
fi

if [ $stage -le 9 ]; then
  ### Triphone + LDA and MLLT + SAT and FMLLR
  # Training
  echo "Starting SAT+FMLLR training."
  steps/align_si.sh --nj $nj --cmd "$cmd" \
      --use-graphs true data/train data/lang exp/tri4b exp/tri4b_ali
  steps/train_sat.sh --cmd "$cmd" 4500 50000 \
      data/train data/lang exp/tri4b_ali exp/tri5b
  echo "SAT+FMLLR training done."

  (
  echo "Decoding the dev set using SAT+FMLLR models."
  utils/mkgraph.sh data/lang_test  exp/tri5b exp/tri5b/graph
  steps/decode_fmllr.sh --nj $dev_nj --cmd "$cmd" \
      exp/tri5b/graph  data/dev exp/tri5b/decode_dev

  echo "SAT+FMLLR decoding done."
  ) &
fi

if [ $stage -le 10 ]; then
  ### Triphone + LDA and MLLT + SAT and FMLLR
  # Training
  echo "Starting SAT+FMLLR training."
  steps/align_si.sh --nj $nj --cmd "$cmd" \
      --use-graphs true data/train data/lang exp/tri5b exp/tri5b_ali
  steps/train_sat.sh --cmd "$cmd" 4500 50000 \
      data/train data/lang exp/tri5b_ali exp/tri6b
  echo "SAT+FMLLR training done."

  (
  echo "Decoding the dev set using SAT+FMLLR models."
  utils/mkgraph.sh data/lang_test  exp/tri6b exp/tri6b/graph
  steps/decode_fmllr_extra.sh --nj $dev_nj --cmd "$cmd" \
      exp/tri6b/graph  data/dev exp/tri6b/decode_dev

  echo "SAT+FMLLR decoding done."
  ) &
fi

if [ $stage -le 11 ]; then
  ### Triphone + LDA and MLLT + SAT and FMLLR
  # Training
  echo "Starting SAT+FMLLR training."
  steps/align_si.sh --nj $nj --cmd "$cmd" \
      --use-graphs true data/train data/lang exp/tri6b exp/tri6b_ali
  steps/train_sat.sh --cmd "$cmd" 4500 50000 \
      data/train data/lang exp/tri6b_ali exp/tri7b
  echo "SAT+FMLLR training done."

  (
  echo "Decoding the dev set using SAT+FMLLR models."
  utils/mkgraph.sh data/lang_test  exp/tri7b exp/tri7b/graph
  steps/decode_fmllr_extra.sh --nj $dev_nj --cmd "$cmd" \
      exp/tri7b/graph  data/dev exp/tri7b/decode_dev

  echo "SAT+FMLLR decoding done."
  ) &
fi

if [ $stage -le 12 ]; then
  ### Triphone + LDA and MLLT + SAT and FMLLR
  # Training
  echo "Starting SAT+FMLLR training."
  steps/align_si.sh --nj $nj --cmd "$cmd" \
      --use-graphs true data/train data/lang exp/tri7b exp/tri7b_ali
  steps/train_sat.sh --cmd "$cmd" 4500 50000 \
      data/train data/lang exp/tri7b_ali exp/tri8b
  echo "SAT+FMLLR training done."

  (
  echo "Decoding the dev set using SAT+FMLLR models."
  utils/mkgraph.sh data/lang_test  exp/tri8b exp/tri8b/graph
  steps/decode_fmllr_extra.sh --nj $dev_nj --cmd "$cmd" \
      exp/tri8b/graph  data/dev exp/tri8b/decode_dev

  echo "SAT+FMLLR decoding done."
  ) &
fi

dev_nj=32
if [ $stage -le 13 ]; then
  ### Triphone + LDA and MLLT + SAT and FMLLR
  # Training
  echo "Starting SAT+FMLLR training."
  steps/cleanup/segment_long_utterances.sh --nj 64  \
    --cmd "$cmd" exp/tri7b data/lang_test/ data/train data/train.seg exp/segment_train
  steps/align_si.sh --nj $nj --cmd "$cmd" \
      --use-graphs false data/train.seg/ data/lang exp/tri8b exp/tri8b_ali.seg
  steps/train_sat.sh --cmd "$cmd" 4500 50000 \
      data/train.seg/ data/lang exp/tri8b_ali.seg exp/tri9b.seg
  echo "SAT+FMLLR training done."

  (
  echo "Decoding the dev set using SAT+FMLLR models."
  utils/mkgraph.sh data/lang_test  exp/tri9b.seg exp/tri9b.seg/graph
  steps/decode_fmllr_extra.sh --nj $dev_nj --cmd "$cmd" \
      exp/tri9b.seg/graph  data/dev exp/tri9b.seg/decode_dev

  echo "SAT+FMLLR decoding done."
  ) &
fi


if [ $stage -le 14 ]; then
  echo "Starting SGMM training."
  steps/align_fmllr.sh --nj $nj --cmd "$cmd" \
      data/train.seg data/lang exp/tri9b.seg/ exp/tri9b_ali

  steps/train_ubm.sh --cmd "$cmd"  \
      600 data/train.seg data/lang exp/tri9b_ali exp/ubm

  steps/train_sgmm2.sh --cmd "$cmd"  \
       5200 12000 data/train.seg data/lang exp/tri9b_ali exp/ubm/final.ubm exp/sgmm2
  echo "SGMM training done."

  (
  echo "Decoding the dev set using SGMM models"
  # Graph compilation
  utils/mkgraph.sh data/lang_test exp/sgmm2 exp/sgmm2/graph

  steps/decode_sgmm2.sh --nj $dev_nj --cmd "$cmd" \
      --transform-dir exp/tri9b.seg/decode_dev \
      exp/sgmm2/graph data/dev exp/sgmm2/decode_dev

  ) &
fi

nj=16
if [ $stage -le 15 ]; then
  echo "Starting second SGMM training."
  #steps/align_sgmm2.sh   --transform-dir exp/tri9b.seg/ --cmd $cmd --nj $nj \
  #  data/train.seg data/lang exp/sgmm2/ exp/sgmm2_ali
  #steps/get_prons.sh --cmd "$cmd" data/train.seg data/lang exp/sgmm2_ali

  #utils/dict_dir_add_pronprobs.sh --max-normalize true data/local/dict \
  #  exp/sgmm2_ali/pron_counts_nowb.txt exp/sgmm2_ali/sil_counts_nowb.txt \
  #  exp/sgmm2_ali/pron_bigram_counts_nowb.txt data/local/dict_pp

  #utils/lang/make_unk_lm.sh data/local/dict exp/make_unk

  utils/prepare_lang.sh  --phone-symbol-table data/lang/phones.txt \
    --unk-fst exp/make_unk/unk_fst.txt \
    data/local/dict "<unk>" data/local/dict_tmp data/lang_unk

  #utils/prepare_lang.sh  --phone-symbol-table data/lang/phones.txt \
  #  --unk-fst exp/make_unk/unk_fst.txt \
  #  data/local/dict_pp "<unk>" data/local/dict_pp_tmp data/lang_pp

  cp -R data/lang_unk/* data/lang_unk_test/
  cp data/lang_test/G.fst data/lang_unk_test

  (
  echo "Decoding the dev set using SGMM models"
  # Graph compilation
  utils/mkgraph.sh data/lang_unk_test exp/sgmm2 exp/sgmm2/graph_unk

  steps/decode_sgmm2.sh --nj $dev_nj --cmd "$cmd" \
      --transform-dir exp/tri9b.seg/decode_dev \
      exp/sgmm2/graph_unk data/dev exp/sgmm2/decode_dev_unk

  ) &
fi
wait
