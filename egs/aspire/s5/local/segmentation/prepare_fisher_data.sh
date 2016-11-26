#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

. path.sh
. cmd.sh

cat <<EOF > $dir/fisher_sad.map
sil 0
sil_B 0
sil_E 0
sil_I 0
sil_S 0
laughter 2
laughter_B 2
laughter_E 2
laughter_I 2
laughter_S 2
noise 2
noise_B 2
noise_E 2
noise_I 2
noise_S 2
oov 3
oov_B 3
oov_E 3
oov_I 3
oov_S 3
EOF

# Expecting the user to have done run.sh
local/segmentation/prepare_unsad_data.sh \
  --sad-map $dir/fisher_sad.map \
  --config-dir conf \
  --reco-nj 40 --nj 100 \
  --sat-model exp/tri4a \
  --lang-test data/lang_test \
  data/fisher_train_100k \
  data/lang \
  exp/tri3a_ali exp/tri3a \
  exp/unsad/make_unsad_fisher_train_100k 

data_dir=data/fisher_train_100k_whole

if [ ! -z $subset ]; then
  # Work on a subset
  utils/subset_data_dir.sh ${data_dir} $subset \
    ${data_dir}_$subset
  data_dir=${data_dir}_$subset
fi

local/segmentation/run_corrupt.sh \
  --data-dir $data_dir \
  --reco-vad-dir exp/unsad/make_unsad_fisher_train_100k/tri3a_reco_vad_fisher_train_100k_sp/ \
  --feat-suffix hires_bp --mfcc-config conf/mfcc_hires_bp.conf 

local/segmentation/run_corrupt_music.sh \
  --data-dir $data_dir \
  --reco-vad-dir exp/unsad/make_unsad_fisher_train_100k/tri3a_reco_vad_fisher_train_100k_sp/ \
  --feat-suffix hires_bp --mfcc-config conf/mfcc_hires_bp.conf
