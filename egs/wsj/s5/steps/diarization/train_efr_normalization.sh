#! /bin/bash 

num_iters=3
cmd=run.pl

set -e -o pipefail
. utils/parse_options.sh
. path.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <data> <ivector-dir> <dir>"
  exit 1
fi

data=$1
ivecdir=$2
dir=$3

for f in $data/utt2spk $ivecdir/ivector.scp; do
  [ ! -f $f ] && echo "$0: Could not find $f" && exit 1
done

mkdir -p $dir

#ivectors="ark:utils/filter_scp.pl $data/utt2spk $ivecdir/ivector.scp | ivector-subtract-global-mean $ivecdir/mean.vec scp:- ark:- | transform-vec $ivecdir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
ivectors="ark:utils/filter_scp.pl $data/utt2spk $ivecdir/ivector.scp | copy-vector scp:- ark:- |"

for iter in `seq 0 $[num_iters-1]`; do
  $cmd $dir/log/compute_mean_iter$iter.log \
    ivector-mean "$ivectors" $dir/mean_iter$iter.vec
  ivectors="$ivectors ivector-subtract-global-mean $dir/mean_iter$iter.vec ark:- ark:- |"
  $cmd $dir/log/compute_transform_iter$iter.log \
    est-pca --read-vectors=true --normalize-mean=false --normalize-variance=true "$ivectors" $dir/transform_iter$iter.mat
  ivectors="$ivectors transform-vec $dir/transform_iter$iter.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
done

echo $num_iters > $dir/num_iters

