#! /bin/bash

. utils/parse_options.sh

cmd=queue.pl

# Final segmentation options
max_segment_length=1000
overlap_length=100

data=$1
ivector_dir=$2
cluster_dir=$3
out_data=$4

if [ $# -ne 4 ]; then
  echo "Usage: $0 <data> <ivector-dir> <cluster-dir> <out-data>"
  exit 1
fi

rm -r $out_data || true
mkdir -p $out_data

for f in wav.scp reco2file_and_channel glm stm; do
  cp $data/$f $out_data
done

utils/data/get_reco2utt.sh $ivector_dir

$cmd $cluster_dir/log/get_segments.log \
  segmentation-init-from-segments --shift-to-zero=false --frame-overlap=0.0 \
  --utt2label-rspecifier=ark,t:$cluster_dir/labels \
  $ivector_dir/segments ark:- \| \
  segmentation-combine-segments-to-recordings ark:- \
  ark,t:$ivector_dir/reco2utt \
  ark:- \| \
  segmentation-post-process --merge-adjacent-segments ark:- ark:- \| \
  segmentation-split-segments \
  --max-segment-length=$max_segment_length \
  --overlap-length=$overlap_length ark:- ark:- \| \
  segmentation-to-segments --frame-overlap=0.0 ark:- \
  ark,t:$out_data/utt2spk $out_data/segments

utils/utt2spk_to_spk2utt.pl $out_data/utt2spk > $out_data/spk2utt
utils/fix_data_dir.sh $out_data
