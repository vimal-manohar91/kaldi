#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This script, called by ../run.sh, creates the HUB4 Broadcast News
# data directory. The required datasets can be found at:
#   https://catalog.ldc.upenn.edu/LDC97S44
#   https://catalog.ldc.upenn.edu/LDC97T22

set -e
tmp_dir=data/local/bn.tmp
remove_overlapping_segments=true

# These parameters are used when refining the annotations.
# A higher frames_per_second provides better resolution at the
# frame boundaries. Set min_seg to control the minimum length of the
# final segments. It seems that the original annotations for segments
# below half a second are not very accurate, so we test only on segments
# longer than this.
frames_per_sec=100
min_seg=0.5

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <sph-dir> <transcript-dir> <data-dir>"
  echo " e.g.: $0 /export/corpora/NIST/96HUB4/h4eng_sp /export/corpora/LDC/LDC97T22/ data"
  exit 1
fi

sph_dir=$1
transcript_dir=$2
data_dir=$3

rm -r $tmp_dir 2>/dev/null || true
mkdir -p $tmp_dir

echo "$0: preparing annotations..."
local/make_annotations_bn.py ${transcript_dir} ${tmp_dir}

if $remove_overlapping_segments; then
  echo "$0: Removing overlapping annotations..."
  local/refine_annotations_bn.py ${tmp_dir} ${frames_per_sec} ${min_seg}
fi

echo "$0: Preparing broadcast news data directories ${data_dir}..."
local/make_bn.py --overlapping-segments-removed=$remove_overlapping_segments ${sph_dir} ${tmp_dir}

mkdir -p ${data_dir}
cp ${tmp_dir}/wav.scp ${data_dir}
cp ${tmp_dir}/utt2spk ${data_dir}
cp ${tmp_dir}/segments ${data_dir}
utils/fix_data_dir.sh ${data_dir}
