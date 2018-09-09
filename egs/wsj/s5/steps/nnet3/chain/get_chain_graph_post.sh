#! /bin/bash

# Copyright 2018  Vimal Manohar
# Apache 2.0

fst_scale=0.5
acwt=0.1
cmd=run.pl

echo $*

. ./cmd.sh
. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <chain-dir> <lat-dir> <dir>"
  echo " e.g.: $0 exp/chain/tdnn exp/chain/tri5_lats exp/chain/tdnn/egs"
  exit 1
fi

chaindir=$1
latdir=$2
dir=$3

nj=$(cat $latdir/num_jobs) || exit 1

lats_rspecifier="ark:gunzip -c $latdir/lat.JOB.gz |"

$cmd JOB=1:$nj $dir/log/get_post.JOB.log \
  chain-lattice-to-post --acoustic-scale=$acwt --fst-scale=$fst_scale \
    $chaindir/den.fst $chaindir/0.trans_mdl "$lats_rspecifier" \
    ark,scp:$dir/numerator_post.JOB.ark,$dir/numerator_post.JOB.scp || exit 1

for n in $(seq $nj); do
  cat $dir/numerator_post.$n.scp
done > $dir/numerator_post.scp
