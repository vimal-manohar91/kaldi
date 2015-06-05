#!/bin/bash

set -o pipefail

. path.sh

cmd=run.pl
nj=4
stage=-1

. parse_options.sh

if [ $# -lt 3 ]; then
  echo "Usage: filter_lattices.sh <decode-dir> [<decode-dir-2> ... <decode-dir-N>] <subset-data-dir> <subset-decode-dir-output>"
  echo " e.g.: filter_lattices.sh exp/tri5/decode_unsup_100k data/unsup_100k_250k exp/tri5/decode_unsup_100k_250k"
  exit 1
fi

decode_dirs=( $@ )
unset decode_dirs[${#decode_dirs[@]}-1]
unset decode_dirs[${#decode_dirs[@]}-1]
subset_data_dir=${@: (-2): 1}
subset_decode_dir=${@: (-1) : 1}

num_decode_dirs=${#decode_dirs[@]}

mkdir -p $subset_decode_dir
echo $nj > $subset_decode_dir/num_jobs
utils/split_data.sh $subset_data_dir $nj || exit 1

job_start=0
for i in `seq 0 $[num_decode_dirs-1]`; do
  decode_dir=${decode_dirs[$i]}
  src_nj=`cat $decode_dir/num_jobs` || exit 1

  if [ $stage -le $i ]; then
    for n in `seq $nj`; do
      $cmd JOB=1:$src_nj $subset_decode_dir/log/filter_lattices.$n.JOB.log \
        lattice-copy --ignore-missing=true --include=$subset_data_dir/split$nj/$n/segments \
        "ark,s,cs:gunzip -c $decode_dir/lat.JOB.gz |" \
        "ark:| gzip -c > $subset_decode_dir/lat.$n.\$[JOB + $job_start].gz" || exit 1
    done
  fi
  job_start=$[job_start+src_nj]
done
    
for n in `seq $nj`; do
  cat $(eval echo $subset_decode_dir/lat.$n.{`seq -s ',' $job_start`}.gz) > $subset_decode_dir/lat.$n.gz || exit 1
done

