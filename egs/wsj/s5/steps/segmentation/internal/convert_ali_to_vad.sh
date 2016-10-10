#! /bin/bash

set -o pipefail
set -e
set -u

. path.sh

cmd=run.pl

frame_shift=0.01
frame_subsampling_factor=1

. parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 data/train exp/tri3_ali data/lang/phones/sad.map exp/tri3_ali_vad"
  exit 1
fi

data=$1
ali_dir=$2
phone_map=$3
dir=$4

for f in $data/segments $phone_map $ali_dir/ali.1.gz; do 
  [ ! -f $f ] && echo "$0: Could not find $f" && exit 1
done

mkdir -p $dir

nj=`cat $ali_dir/num_jobs` || exit 1
echo $nj > $dir/num_jobs

utils/split_data.sh $data $nj

if [ -f $ali_dir/frame_subsampling_factor ]; then
  frame_subsampling_factor=`cat $ali_dir/frame_subsampling_factor`
fi

ali_frame_shift=`perl -e "print ($frame_shift * $frame_subsampling_factor);"`
ali_frame_overlap=`perl -e "print ($ali_frame_shift * 1.5);"`

$cmd JOB=1:$nj $dir/log/get_sad.JOB.log \
  segmentation-init-from-ali \
  "ark:gunzip -c ${ali_dir}/ali.JOB.gz | ali-to-phones --per-frame ${ali_dir}/final.mdl ark:- ark:- |" \
  ark:- \| segmentation-copy --label-map=$phone_map ark:- ark:- \| \
  segmentation-post-process --merge-adjacent-segments ark:- ark,scp:$dir/sad_seg.JOB.ark,$dir/sad_seg.JOB.scp

for n in `seq $nj`; do 
  cat $dir/sad_seg.$n.scp
done | sort -k1,1 > $dir/sad_seg.scp
