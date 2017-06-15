#! /bin/bash

# Copyright 2015-16  Vimal Manohar
# Apache 2.0.

# This script creates sub-segments of an input kaldi data directory using
# SAD alignments and post-processing them.

set -e -o pipefail -u
. path.sh

cmd=run.pl
stage=-10

segmentation_config=conf/segmentation.conf    # Config for post-processing
nj=18

frame_subsampling_factor=1  # The frame-subampling-factor used to create the 
                            # alignments. Usually this would be 3.
frame_shift=0.01  # Used to convert frame numbers to times of segments
frame_overlap=0.015   # Additional padding added to the segments to account 
                      # for snipping during feature extraction

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "This script creates sub-segments of an input kaldi data directory using "
  echo "SAD alignments and post-processing them."
  echo "Usage: $0 <data-dir> <vad-dir> <segmentation-dir> <segmented-data-dir>"
  echo " e.g.: $0 data/dev_aspire_whole exp/vad_dev_aspire data/dev_aspire_seg"
  exit 1
fi

data_dir=$1
vad_dir=$2    # Alignment directory containing frame-level labels
dir=$3
segmented_data_dir=$4

mkdir -p $dir

for f in $vad_dir/ali.1.gz $vad_dir/num_jobs; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f" && exit 1
  fi
done

nj=`cat $vad_dir/num_jobs` || exit 1
utils/split_data.sh $data_dir $nj

if [ $stage -le 0 ]; then
  # Convert the original frame-level SAD into segmentation format
  $cmd JOB=1:$nj $dir/log/segmentation.JOB.log \
    segmentation-init-from-ali \
      "ark:gunzip -c $vad_dir/ali.JOB.gz |" ark:- \| \
    segmentation-copy \
      --frame-subsampling-factor=$frame_subsampling_factor ark:- \
    "ark:| gzip -c > $dir/orig_segmentation.JOB.gz"
fi

echo $nj > $dir/num_jobs

# Create a temporary directory into which we can create the new segments
# file.
if [ $stage -le 1 ]; then
  rm -r $dir/tmp || true
  utils/data/convert_data_dir_to_whole.sh $data_dir $dir/tmp
  rm $dir/tmp || true
fi

if [ $stage -le 2 ]; then
  # --frame-overlap is set to 0 to not do any additional padding when writing
  # segments. This padding will be done later by the option
  # --segment-end-padding to utils/data/subsegment_data_dir.sh.
  steps/segmentation/internal/post_process_segments.sh \
    --stage $stage --cmd "$cmd" \
    --config $segmentation_config --frame-shift $frame_shift \
    --frame-overlap 0 \
    $data_dir $dir $dir/tmp
fi

utils/data/subsegment_data_dir.sh \
  --segment-end-padding $frame_overlap \
  $data_dir $dir/tmp/segments $segmented_data_dir
utils/fix_data_dir.sh $segmented_data_dir

if [ ! -s $segmented_data_dir/utt2spk ] || [ ! -s $segmented_data_dir/segments ]; then
  echo "$0: Segmentation failed to generate segments or utt2spk!"
  exit 1
fi

