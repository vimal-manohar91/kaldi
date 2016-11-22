#! /bin/bash

# Copyright 2015  Vimal Manohar
# Apache 2.0.

set -e -o pipefail -u
. path.sh

cmd=run.pl
stage=-10

# General segmentation options
pad_length=50          # Pad speech segments by this many frames on either side
max_relabel_length=10  # Maximum duration of speech that will be removed as part
                       # of smoothing process. This is only if there are no other
                       # speech segments nearby.
max_intersegment_length=50  # Merge nearby speech segments if the silence
                            # between them is less than this many frames.
post_pad_length=50        # Pad speech segments by this many frames on either side
                          # after the merging process using max_intersegment_length
max_segment_length=1000   # Segments that are longer than this are split into
                          # overlapping frames.
overlap_length=100        # Overlapping frames when segments are split.
                          # See the above option.
min_silence_length=30     # Min silence length at which to split very long segments

frame_shift=0.01
ali_suffix=_acwt0.1

phone_map=

. utils/parse_options.sh

if [ $# -ne 5 ]; then
  echo "Usage: $0 <data-dir> <lang> <vad-dir> <segmentation-dir> <segmented-data-dir>"
  echo " e.g.: $0 data/dev_aspire_whole exp/vad_dev_aspire data/dev_aspire_seg"
  exit 1
fi

data_dir=$1
lang=$2
vad_dir=$3
dir=$4
segmented_data_dir=$5

nj=`cat $vad_dir/num_jobs` || exit 1

if [ $stage -le 0 ]; then
  rm -r $segmented_data_dir || true
  utils/data/convert_data_dir_to_whole.sh $data_dir $segmented_data_dir || exit 1
fi

cat $data_dir/segments | awk '{print $1" "$2}' | \
  utils/utt2spk_to_spk2utt.pl > $data_dir/reco2utt

mkdir -p $dir

if [ -z "$phone_map" ]; then
  phone_map=$dir/phone_map

  {
  cat $lang/phones/silence.int | awk '{print $1" 0"}';
  cat $lang/phones/nonsilence.int | awk '{print $1" 1"}';
  } | sort -k1,1 -n > $dir/phone_map
fi

if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/segmentation.JOB.log \
    segmentation-init-from-ali --reco2utt-rspecifier="ark,t:$data_dir/reco2utt" \
    --segments-rspecifier="ark,t:$data_dir/segments" --frame-shift=$frame_shift \
    "ark:gunzip -c $vad_dir/ali${ali_suffix}.JOB.gz |" ark:- \| \
    segmentation-copy --label-map=$phone_map ark:- ark:$dir/orig_segmentation.JOB 
  
  $cmd JOB=1:$nj $dir/log/post_process_segmentation.JOB.log \
    segmentation-post-process --remove-labels=0 --widen-label=1 --widen-length=$pad_length \
    ark:$dir/orig_segmentation.JOB ark:- \| \
    segmentation-post-process --merge-adjacent-segments=true --max-intersegment-length=$max_intersegment_length ark:- ark:- \| \
    segmentation-post-process --max-relabel-length=$max_relabel_length --relabel-short-segments-class=1 ark:- ark:- \| \
    segmentation-post-process --widen-label=1 --widen-length=$post_pad_length ark:- ark:- \| \
    segmentation-split-segments --alignments="ark:segmentation-to-ali ark:$dir/orig_segmentation.JOB ark:- |" \
    --max-segment-length=$max_segment_length --min-alignment-segment-length=$min_silence_length --ali-label=0 ark:- ark:- \| \
    segmentation-split-segments \
    --max-segment-length=$max_segment_length --overlap-length=$overlap_length ark:- ark:- \| \
    segmentation-to-segments --frame-shift=$frame_shift ark:- \
    ark,t:$dir/utt2spk.JOB \
    ark,t:$dir/segments.JOB || exit 1
fi

for n in `seq $nj`; do
  cat $dir/utt2spk.$n
done > $segmented_data_dir/utt2spk

for n in `seq $nj`; do
  cat $dir/segments.$n
done > $segmented_data_dir/segments

if [ ! -s $segmented_data_dir/utt2spk ] || [ ! -s $segmented_data_dir/segments ]; then
  echo "$0: Segmentation failed to generate segments or utt2spk!"
  exit 1
fi

utils/utt2spk_to_spk2utt.pl $segmented_data_dir/utt2spk > $segmented_data_dir/spk2utt || exit 1
utils/fix_data_dir.sh $segmented_data_dir

if [ ! -s $segmented_data_dir/utt2spk ] || [ ! -s $segmented_data_dir/segments ]; then
  echo "$0: Segmentation failed to generate segments or utt2spk!"
  exit 1
fi
