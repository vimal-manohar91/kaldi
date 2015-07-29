#!/bin/bash

# Copyright 2015  Vimal Manohar
# Apache 2.0

# This example does multilingual online nnet2 training with i-vectors
# Eg: ./local/online/run_nnet2_multilang_common.sh --l=ASM --ali=exp/ASM/tri5_ali --data=data/ASM/train \
#											       --l=BNG --ali=exp/BNG/tri5_ali --data=data/BNG/train \
#												   --l=CNT --ali=exp/CNT/tri5_ali --data=data/CNT/train

. ./path.sh
. ./cmd.sh

stage=-10

. utils/parse_options.sh

set -u
set -e 
set -o pipefail

nlangs=0
j=0
while [ $# -gt 0 ]; do
  lang[j]=$1
  ali[j]=$2
  dataid[j]=$3

  shift; shift; shift;
  nlangs=$[nlangs+1]
  j=$nlangs
done
nlangs=$[nlangs-1]

# Check if all the user i/p directories exist
for i in  $(seq 0 $nlangs)
do
	echo "lang = ${lang[i]}, alidir = ${ali[i]}, dataid = ${dataid[i]}"
	[ ! -e ${ali[i]} ] && echo  "Missing  ${ali[i]}" && exit 1
	[ ! -e data/${dataid[i]} ] && echo "Missing data/${dataid[i]}" && exit 1
done

# Make the features
data_multilang=data_multi/train

if [ $stage -le 0 ]; then
  for i in `seq 0 $nlangs`; do
    echo "Language = ${lang[i]}: Generating features from datadir = data/${dataid[i]}"

    this_data=$data_multilang/${lang[i]}
    gmm_ali=${ali[i]}
    data_id=${dataid[i]}	

    mfccdir=mfcc_hires/$data_id
    utils/copy_data_dir.sh data/$data_id ${this_data}_hires
    steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" ${this_data}_hires exp/make_hires/$data_id $mfccdir
    steps/compute_cmvn_stats.sh ${this_data}_hires exp/make_hires/$data_id $mfccdir || exit 1

  done
fi

if [ $stage -le 2 ]; then
  # We need to build a small system to train diag-UBM on top of.
  # We use only the first language for this.
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
    data/${dataid[0]} data/${lang[0]}/lang exp/${lang[0]}/tri4 exp/nnet2_online/${lang[0]}/tri4_ali
fi

if [ $stage -le 3 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --realign-iters "" \
    --splice-opts "--left-context=3 --right-context=3" \
    2500 36000 $data_multilang/${lang[0]}_hires data/${lang[0]}/lang \
     exp/nnet2_online/${lang[0]}/tri4_ali exp/nnet2_online/${lang[0]}/tri4b
fi

if [ $stage -le 4 ]; then
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
    --num-threads 6 --num-frames 400000 $data_multilang/${lang[0]}_hires 256 \
    exp/nnet2_online/${lang[0]}/tri4b exp/nnet2_online/${lang[0]}/diag_ubm
fi

if [ $stage -le 5 ]; then
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    $data_multilang/${lang[0]}_hires exp/nnet2_online/${lang[0]}/diag_ubm \
    exp/nnet2_online/${lang[0]}/extractor || exit 1
fi

if [ $stage -le 6 ]; then

  for i in `seq 0 $nlangs`; do
    echo "Language = ${lang[i]}: Extracting i-vectors"
    
    this_data=$data_multilang/${lang[i]} 
    data_id=${dataid[i]}	

    steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 $data_multilang/${lang[i]}_hires $data_multilang/${lang[i]}_hires_max2

    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
      $data_multilang/${lang[i]}_hires_max2 exp/nnet2_online/${lang[0]}/extractor \
      exp/nnet2_online/${lang[i]}/ivectors_train || exit 1
  done
fi

