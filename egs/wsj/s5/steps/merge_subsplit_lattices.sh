#! /bin/bash

cmd=run.pl

. ./path.sh
. utils/parse_options.sh

if [ $# -ne 1 ]; then
  exit 1
fi

dir=$1

nj=$(cat $dir/num_jobs)
sub_split=$(cat $dir/sub_split) || exit 1

$cmd JOB=1:$nj $dir/log/merge_subsplits.JOB.log \
  for s in $(seq $sub_split)\; do gunzip -c $dir/lat.JOB.\$s.gz\; done \| gzip -c '>' $dir/lat.JOB.gz

rm $dir/sub_split
