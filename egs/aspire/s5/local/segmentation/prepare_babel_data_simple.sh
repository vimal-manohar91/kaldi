#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script prepares Babel data for training speech activity detection,
# music detection.

. path.sh
. cmd.sh

set -e
set -o pipefail
set -u

lang_id=cantonese_flp_simple
subset_fraction=0.1
realign=false

# All the paths below can be modified to any absolute path.
ROOT_DIR=/export/b17/jtrmal/babel/101-cantonese-flp-p-basic

stage=-1

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  echo "This script is to serve as an example recipe."
  echo "Edit the script to change variables if needed."
  exit 1
fi

dir=exp/unsad_simple/make_unsad_babel_${lang_id}_train_cleaned_pitch_sp   # Work dir
train_data_dir=$ROOT_DIR/data/train_cleaned_pitch_sp
unperturbed_data_dir=$ROOT_DIR/data/train_cleaned_pitch
model_dir=$ROOT_DIR/exp/tri5_cleaned   # Model directory
lang=$ROOT_DIR/data/lang  # Language directory

mkdir -p $dir

# Hard code the mapping from phones to SAD labels
# 0 for silence, 1 for speech, 2 for noise, 3 for unk
cat <<EOF > $dir/babel_sad.map
<oov> 3
<oov>_B 3
<oov>_E 3
<oov>_I 3
<oov>_S 3
<sss> 2
<sss>_B 2
<sss>_E 2
<sss>_I 2
<sss>_S 2
<vns> 2
<vns>_B 2
<vns>_E 2
<vns>_I 2
<vns>_S 2
SIL 0
SIL_B 0
SIL_E 0
SIL_I 0
SIL_S 0
EOF

utils/copy_data_dir.sh $train_data_dir data/babel_${lang_id}_train_sp
train_data_dir=data/babel_${lang_id}_train_sp

utils/copy_data_dir.sh $unperturbed_data_dir data/babel_${lang_id}_train
unperturbed_data_dir=data/babel_${lang_id}_train

if $realign; then
  ali_dir=$dir/`basename $model_dir`_ali_$(basename $train_data_dir)

  steps/align_fmllr.sh --nj 32 --cmd "$train_cmd" \
    $train_data_dir $lang $model_dir $ali_dir

  # Expecting the user to have done run.sh to have $ali_dir,
  # $lang, $train_data_dir
  local/segmentation/prepare_unsad_data_simple.sh \
    --sad-map $dir/babel_sad.map --cmd "$train_cmd" \
    $train_data_dir $lang $ali_dir $dir
  
  vad_dir=$dir/`basename $ali_dir`_vad_$(basename $train_data_dir)
else
  local/segmentation/prepare_unsad_data_simple.sh --speed-perturb true \
    --sad-map $dir/babel_sad.map --cmd "$train_cmd" \
    $unperturbed_data_dir $lang $model_dir $dir

  vad_dir=$dir/`basename $model_dir`_vad_$(basename $unperturbed_data_dir)
fi

data_dir=${unperturbed_data_dir}

if [ ! -z "$subset_fraction" ]; then
  # Work on a subset
  num_utts=`cat $unperturbed_data_dir/utt2spk | wc -l`
  subset=`python -c "n=int($num_utts * $subset_fraction / 1000.0) * 1000; print (n if n > 4000 else 4000)"`
  subset_affix=`echo $subset | perl -pe 's/000/k/g'`
  utils/subset_data_dir.sh --speakers ${unperturbed_data_dir} $subset \
    ${unperturbed_data_dir}_${subset_affix}
  data_dir=${unperturbed_data_dir}_${subset_affix}
fi

# Add noise from MUSAN corpus to data directory and create a new data directory
local/segmentation/do_corruption_data_dir_snr.sh \
  --cmd "$train_cmd" --nj 40 \
  --data-dir $data_dir \
  --vad-dir $vad_dir \
  --feat-suffix hires_bp --mfcc-config conf/mfcc_hires_bp.conf 

# Add music from MUSAN corpus to data directory and create a new data directory
local/segmentation/do_corruption_data_dir_music.sh \
  --cmd "$train_cmd" --nj 40 \
  --data-dir $data_dir \
  --vad-dir $vad_dir \
  --feat-suffix hires_bp --mfcc-config conf/mfcc_hires_bp.conf
