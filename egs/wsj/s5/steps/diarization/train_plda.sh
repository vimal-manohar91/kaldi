#! /bin/bash

cmd=run.pl
min_length=0
utts_per_spk_max=50000
utts_per_spk_min=3
do_sph_norm=false
do_efr_norm=false
use_wccn=false
num_iters=3
transform_dir=
stage=0

set -e -o pipefail -u

. path.sh
. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <data> <ivector-dir> <dir>"
  exit 1
fi

data=$1
ivecdir=$2
dir=$3

[ -z "$transform_dir" ] && transform_dir=$ivecdir
for f in $data/spk2utt $data/segments $ivecdir/mean.vec $ivecdir/ivector.scp $transform_dir/transform.mat; do
  if [ ! -f $f ]; then
    echo "$0: Could not find $f"
    exit 1
  fi
done

mkdir -p $dir

if [ $stage -le 0 ]; then
  mkdir -p $dir/tmp
  cp $data/{spk2utt,segments,utt2spk} $dir/tmp || exit 1
  awk '{print $1" "$4-$3}' $data/segments > $dir/tmp/utt2dur
  cp $ivecdir/ivector.scp $dir/tmp/feats.scp

  awk -v min_length=$min_length '{if ($2 > min_length) print $0}' $dir/tmp/utt2dur > $dir/tmp/utt2dur.modified

  if [ ! -s $dir/tmp/utt2dur.modified ]; then
    echo "$0: Filtering utterances greater than $min_length. No utterances remained."
    exit 1
  fi

  cp $dir/tmp/utt2dur.modified $dir/tmp/utt2dur
  mv $dir/tmp/segments{,.tmp}
  utils/filter_scp.pl $dir/tmp/utt2dur $data/utt2spk > $dir/tmp/utt2spk
  utils/utt2spk_to_spk2utt.pl $dir/tmp/utt2spk > $dir/tmp/spk2utt

  python -c 'import sys, random
utts_per_spk_min = int(sys.argv[1])
utts_per_spk_max = int(sys.argv[2])
assert utts_per_spk_max > utts_per_spk_min and utts_per_spk_min >= 2
num_done = 0
for line in sys.stdin.readlines():
  parts = line.strip().split()
  if len(parts) <= utts_per_spk_min:
    continue
  spk = parts[0]
  parts =  parts[1:]
  random.shuffle(parts)
  num_utts = min(utts_per_spk_max, len(parts))
  print(spk + " " + " ".join(parts[0:num_utts]))
  num_done += 1 
if num_done == 0:
  sys.stderr.write("Failed to subselect speakers!\n")
  sys.exit(1)' \
      $utts_per_spk_min $utts_per_spk_max < $dir/tmp/spk2utt > \
      $dir/tmp/spk2utt.modified
  cp $dir/tmp/spk2utt.modified $dir/tmp/spk2utt
  utils/spk2utt_to_utt2spk.pl $dir/tmp/spk2utt > $dir/tmp/utt2spk
  utils/fix_data_dir.sh $dir/tmp
fi


ivectors="ark:utils/filter_scp.pl $dir/tmp/utt2spk $ivecdir/ivector.scp | ivector-subtract-global-mean $ivecdir/mean.vec scp:- ark:- | transform-vec $transform_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"

if [ `readlink -f $ivecdir` != `readlink -f $dir` ]; then
  cp $ivecdir/mean.vec $dir
  cp $transform_dir/transform.mat $dir
fi

if $do_sph_norm; then
  if [ $stage -le 1 ]; then
    steps/diarization/train_spherical_nuisance_normalization.sh --num-iters $num_iters --use-wccn $use_wccn \
      $dir/tmp $ivecdir $dir/snn
  fi
  
  ivectors="ark:utils/filter_scp.pl $dir/tmp/utt2spk $ivecdir/ivector.scp | copy-vector scp:- ark:- |"
  for iter in `seq 0 $[num_iters-1]`; do
    ivectors="$ivectors ivector-subtract-global-mean $dir/snn/mean_iter$iter.vec ark:- ark:- | transform-vec $dir/snn/transform_iter$iter.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
  done
elif $do_efr_norm; then
  steps/diarization/train_efr_normalization.sh --num-iters $num_iters \
    $dir/tmp $ivecdir $dir/efr
  
  ivectors="ark:utils/filter_scp.pl $dir/tmp/utt2spk $ivecdir/ivector.scp | copy-vector scp:- ark:- |"
  for iter in `seq 0 $[num_iters-1]`; do
    ivectors="$ivectors ivector-subtract-global-mean $dir/efr/mean_iter$iter.vec ark:- ark:- | transform-vec $dir/efr/transform_iter$iter.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
  done
fi

if [ $stage -le 2 ]; then
  $cmd $dir/log/plda.log \
    ivector-compute-plda ark:$dir/tmp/spk2utt "$ivectors" \
    $dir/plda
fi
