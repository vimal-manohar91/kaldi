#!/bin/bash

# Copyright   2016  Vimal Manohar
#             2016  Johns Hopkins University (author: Daniel Povey)
#             2017  Nagendra Kumar Goel
# Apache 2.0

# This script demonstrates how to re-segment training data selecting only the
# "good" audio that matches the transcripts.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# biased language model built from the reference, and then work out the
# segmentation from a ctm like file.

# For nnet3 and chain results after cleanup, see the scripts in
# local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh

# GMM Results for speaker-independent (SI) and speaker adaptive training (SAT) systems on dev and test sets
# [will add these later].

set -e
set -o pipefail
set -u

stage=0
cleanup_stage=0
data=data/train_hires
cleanup_affix=cleaned
srcdir=exp/nnet3/tdnn_1g
langdir=data/lang
boost_sil=1.5
nj=100
decode_nj=16
decode_num_threads=4

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh

cleaned_data=${data}_${cleanup_affix}
lores_cleaned_data=$(echo $data | sed 's/_hires/_seg/')_${cleanup_affix}

dir=${srcdir}_${cleanup_affix}_work
cleaned_dir=${srcdir}_${cleanup_affix}

if [ $stage -le 1 ]; then
  # This does the actual data cleanup.
  steps/cleanup/clean_and_segment_data_nnet3.sh  --stage $cleanup_stage \
    --cmd "$cmd" --nj "$nj" \
    $data  $langdir $srcdir $dir $cleaned_data
fi

if [ $stage -le 2 ]; then
  #utils/copy_data_dir.sh $cleaned_data $lores_cleaned_data
  #steps/make_mfcc.sh --cmd "$cmd" --nj 32 $lores_cleaned_data
  #steps/compute_cmvn_stats.sh $lores_cleaned_data
  steps/align_fmllr.sh --nj $nj --cmd "$cmd" \
    $lores_cleaned_data data/lang/ exp/tri9b.seg/ ${srcdir}_ali_${cleanup_affix}
fi

if [ $stage -le 3 ]; then
  steps/train_sat.sh --cmd "$cmd" --boost-silence $boost_sil \
    4500 50000 $lores_cleaned_data $langdir ${srcdir}_ali_${cleanup_affix} ${cleaned_dir}
fi
