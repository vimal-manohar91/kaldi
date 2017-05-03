#! /bin/bash

. utils/parse_options.sh

cmd=queue.pl

# Final segmentation options
max_segment_length=1000
overlap_length=100

data=$1
cluster_dir=$2
out_data=$3

if [ $# -ne 3 ]; then
  echo "Usage: $0 <data> <cluster-dir> <out-data>"
  exit 1
fi

rm -r $out_data || true
mkdir -p $out_data

for f in wav.scp reco2file_and_channel glm stm; do
  [ -f $f ] && cp $data/$f $out_data
done

utils/data/get_reco2utt.sh $data

$cmd $cluster_dir/log/get_segments.log \
  segmentation-init-from-segments --shift-to-zero=false --frame-overlap=0.0 \
  --utt2label-rspecifier=ark,t:$cluster_dir/labels \
  $data/segments ark:- \| \
  segmentation-combine-segments-to-recordings ark:- \
  ark,t:$data/reco2utt \
  ark:- \| \
  segmentation-post-process --merge-adjacent-segments ark:- ark:- \| \
  segmentation-split-segments \
  --max-segment-length=$max_segment_length \
  --overlap-length=$overlap_length ark:- ark:- \| \
  segmentation-to-segments --frame-overlap=0.0 ark:- \
  ark,t:$out_data/utt2spk $out_data/segments

utils/utt2spk_to_spk2utt.pl $out_data/utt2spk > $out_data/spk2utt
utils/fix_data_dir.sh $out_data
