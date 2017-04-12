#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

nj=4
cmd=queue.pl

do_change_point_detection=true
change_point_split_opts="--use-full-covar --distance-metric=glr"
change_point_merge_opts="--use-full-covar --distance-metric=bic --bic-penalty=5.0"
frame_overlap=0.015

max_blend_length=10
max_intersegment_length=10

num_frames=150
num_frames_overlap=75
stage=0

. path.sh

set -e -o pipefail -u

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <whole-data> <data> <dir> <out-data>"
  echo " e.g.: $0 data/eval97.seg_whole data/eval97.seg exp/change_point/eval97.seg data/eval97.seg_uniform"
  exit 1
fi

data_whole=$1
data=$2
dir=$3
out_data=$4

utils/split_data.sh --per-reco $data_whole $nj

mkdir -p $out_data

if $do_change_point_detection; then
  if [ $stage -le 1 ]; then
    $cmd JOB=1:$nj $dir/log/split_by_change_points.JOB.log \
      segmentation-init-from-segments --frame-overlap=$frame_overlap \
        $data_whole/split${nj}reco/JOB/segments ark:- \| \
      segmentation-split-by-change-points $change_point_split_opts ark:- \
        scp:$data_whole/split${nj}reco/JOB/feats.scp ark:- \| \
      segmentation-cluster-adjacent-segments $change_point_merge_opts \
        ark:- scp:$data_whole/split${nj}reco/JOB/feats.scp ark:- \| \
      segmentation-post-process --merge-adjacent-segments ark:- ark:$dir/temp_segmentation.JOB.ark
  fi
  if [ $stage -le 2 ]; then
    $cmd JOB=1:$nj $dir/log/get_subsegments.JOB.log \
      segmentation-filter-segments \
        ark:$dir/temp_segmentation.JOB.ark \
        "ark:segmentation-init-from-segments --frame-overlap=0.0 --shift-to-zero=false $data/segments ark:- | segmentation-combine-segments-to-recordings ark:- ark,t:$data/reco2utt ark:- | segmentation-post-process --merge-adjacent-segments ark:- ark:- |" \
        ark:- \| \
      segmentation-post-process --max-blend-length=$max_blend_length \
        --max-intersegment-length=$max_intersegment_length --blend-short-segments-class=1 \
        ark:- ark:- \| \
      segmentation-post-process --remove-labels=0 --merge-adjacent-segments ark:- ark:- \| \
      segmentation-post-process --remove-labels=-1 --max-remove-length=10 ark:- ark:- \| \
      segmentation-to-segments --single-speaker --frame-overlap=0.0 \
        ark:- ark:/dev/null $out_data/sub_segments
  fi

  if [ $stage -le 3 ]; then
    utils/data/subsegment_data_dir.sh ${data_whole} $out_data/sub_segments $out_data
  fi
else
  if [ $stage -le 1 ]; then
  $cmd $dir/log/get_subsegments.log \
    segmentation-init-from-segments --frame-overlap=$frame_overlap $data/segments ark:- \| \
    segmentation-split-segments --max-segment-length=$num_frames --overlap-length=$num_frames_overlap ark:- ark:- \| \
    segmentation-to-segments --frame-overlap=0.0 --single-speaker ark:- ark:/dev/null \
    ${out_data}/sub_segments
  fi

  if [ $stage -le 3 ]; then 
    utils/data/subsegment_data_dir.sh ${data} ${out_data}/sub_segments $out_data
  fi
fi
