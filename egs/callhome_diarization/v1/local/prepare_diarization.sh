#!/bin/bash
# Copyright 2016  David Snyder
#           2017  Vimal Manohar
# Apache 2.0.

cmvn_opts="--norm-means=false --norm-vars=false"
apply_sliding_cmvn=true
num_components=2048
ivector_dim=128
affix=
chunk_size=150
period=150
min_chunk_size=100
ivector_suffix=_150

. cmd.sh
. path.sh

. utils/parse_options.sh

# Reduce the amount of training data for the UBM.
utils/subset_data_dir.sh data/train 16000 data/train_16k
utils/subset_data_dir.sh data/train 32000 data/train_32k

# Train UBM and i-vector extractor.
diarization/train_diag_ubm.sh --cmd "$train_cmd -l mem_free=20G,ram_free=20G" \
  --nj 20 --num-threads 8 --delta-order 1 \
  --cmvn-opts "$cmvn_opts" --apply-sliding-cmvn $apply_sliding_cmvn \
  data/train_16k $num_components \
  exp/diag_ubm${affix}_${num_components}

diarization/train_full_ubm.sh --nj 40 --remove-low-count-gaussians false \
  --cmd "$train_cmd --mem 25G" data/train_32k \
  exp/diag_ubm${affix}_$num_components exp/full_ubm${affix}_$num_components

diarization/train_ivector_extractor.sh \
  --cmd "$train_cmd --mem 35G" \
  --ivector-dim $ivector_dim --num-iters 5 \
  exp/full_ubm${affix}_$num_components/final.ubm data/train \
  exp/extractor${affix}_c${num_components}_i${ivector_dim}

# TODO(vimal): Extract chunks of different sizes to get ivectors 
# with different variances.
diarization/extract_ivectors.sh --cmd "$train_cmd --mem 10G" \
  --nj 40 --use-vad true \
  --chunk-size $chunk_size --period $period \
  --min-chunk-size $min_chunk_size \
  exp/extractor_c${num_components}_i${ivector_dim} \
  data/sre exp/ivectors_sre${ivector_suffix}
