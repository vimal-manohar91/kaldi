#!/bin/bash

utt_prefixes=
max_jobs_run=30
nj=100
cmd=queue.pl
write_compact=true

. ./path.sh
. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <out-data> <src-lat-dir> <out-lat-dir>"
  exit 1
fi

data=$1
src_dir=$2
dir=$3

mkdir -p $dir

num_jobs=$(cat $src_dir/num_jobs)

rm -f $dir/lat_tmp.*.{ark,scp} 2>/dev/null

# Copy the lattices temporarily
$cmd --max-jobs-run $max_jobs_run JOB=1:$num_jobs $dir/log/copy_lattices.JOB.log \
  lattice-copy --write-compact=$write_compact \
  "ark:gunzip -c $src_dir/lat.JOB.gz |" \
  ark,scp:$dir/lat_tmp.JOB.ark,$dir/lat_tmp.JOB.scp || exit 1

if [ ! -z "$utt_prefixes" ]; then
  # Make copies of utterances for perturbed data
  for p in $utt_prefixes; do
    cat $dir/lat_tmp.*.scp | awk -v p=$p '{print p$0}'
  done | sort -k1,1 > $dir/lat_out.scp
else
  cat $dir/lat_tmp.*.scp | sort -k1,1 > $dir/lat_out.scp
fi

utils/split_data.sh ${data} $nj

# Copy and dump the lattices for perturbed data
$cmd --max-jobs-run $max_jobs_run JOB=1:$nj $dir/log/copy_out_lattices.JOB.log \
  lattice-copy --write-compact=$write_compact \
  "scp:utils/filter_scp.pl ${data}/split$nj/JOB/utt2spk $dir/lat_out.scp |" \
  "ark:| gzip -c > $dir/lat.JOB.gz" || exit 1

rm $dir/lat_tmp.* $dir/lat_out.scp

echo $nj > $dir/num_jobs

for f in cmvn_opts final.mdl splice_opts tree frame_subsampling_factor; do
  if [ -f $src_dir/$f ]; then cp $src_dir/$f $dir/$f; fi 
done
