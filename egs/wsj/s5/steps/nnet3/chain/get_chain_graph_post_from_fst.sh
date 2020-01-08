#! /bin/bash

# Copyright 2018  Vimal Manohar
# Apache 2.0

stage=0
cmd=run.pl
nj=4
use_gpu=false
min_post=0.01
fst_scale=1.0
frames_per_chunk=50
iter=final
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
online_ivector_dir=

echo $*

. ./cmd.sh
. utils/parse_options.sh

if [ $# -ne 4 ] && [ $# -ne 3 ]; then
  echo "Usage: <data> <chain-dir> [<fst-dir>] <dir>"
  echo " e.g.: $0 data/train exp/chain/tdnn exp/chain/tdnn/egs"
  exit 1
fi

data=$1
chaindir=$2
if [ $# -eq 4 ]; then
  fstdir=$3
  dir=$4
else
  fstdir=$chaindir
  dir=$3
fi

extra_files=
if [ ! -z "$online_ivector_dir" ]; then
  steps/nnet2/check_ivectors_compatible.sh $chaindir $online_ivector_dir || exit 1
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"
fi

for f in $data/feats.scp $chaindir/tree $chaindir/$iter.mdl $fstdir/tree $fstdir/den.fst $extra_files; do 
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

diff $chaindir/tree $fstdir/tree || { echo "$chaindir/tree and $fstdir/tree differ!" && exit 1; }

sdata=$data/split$nj;
cmvn_opts=`cat $chaindir/cmvn_opts` || exit 1;

num_pdfs=$(tree-info $chaindir/tree |grep num-pdfs|awk '{print $2}')

## Set up features.
echo "$0: feature type is raw"

feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"

if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
fi

frame_subsampling_opt=
if [ -f $chaindir/frame_subsampling_factor ]; then
  # e.g. for 'chain' systems
  frame_subsampling_opt="--frame-subsampling-factor=$(cat $chaindir/frame_subsampling_factor)"
fi

gpu_opt="--use-gpu=no"
gpu_queue_opt=

if $use_gpu; then
  gpu_queue_opt="--gpu 1"
  gpu_opt="--use-gpu=yes"
fi

# note: --compression-method=3 is kTwoByteAuto: Each element is stored in two
# bytes as a uint16, with the representable range of values chosen
# automatically with the minimum and maximum elements of the matrix as its
# edges.
compress_opts="--compress=true --compression-method=3"

if [ $stage -le 1 ]; then
  $cmd $gpu_queue_opt JOB=1:$nj $dir/log/get_post.JOB.log \
    nnet3-chain-compute-post ${gpu_opt} $ivector_opts $frame_subsampling_opt \
    --frames-per-chunk=$frames_per_chunk \
    --extra-left-context=$extra_left_context \
    --extra-right-context=$extra_right_context \
    --extra-left-context-initial=$extra_left_context_initial \
    --extra-right-context-final=$extra_right_context_final \
    $chaindir/$iter.mdl $fstdir/den.fst "$feats" ark:- \| \
  copy-feats $compress_opts ark:- \
    ark,scp:$dir/numerator_post.JOB.ark,$dir/numerator_post.JOB.scp || exit 1

  sleep 5
fi

for n in $(seq $nj); do
  cat $dir/numerator_post.$n.scp
done > $dir/numerator_post.scp
