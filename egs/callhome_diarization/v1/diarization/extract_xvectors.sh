#!/bin/bash

# Copyright     2013  Daniel Povey
#               2016  David Snyder
# Apache 2.0.

# TODO This script extracts iVectors for a set of utterances, given
# features and a trained iVector extractor.  It is based on the
# script sid/extract_ivectors.sh, but uses ivector-extract-dense
# to extract ivectors from overlapping chunks.

# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
left_context=2
right_context=2
chunk_size=150
min_chunk_size=30
pca_dim=
iter=final
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: $0 <nnet-dir> <data> <xvector-dir>"
  echo " e.g.: $0 exp/nnet data/train exp/xvectors"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --num-gselect <n|20>                             # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --min-post <min-post|0.025>                      # Pruning threshold for posteriors"
  echo "  --chunk-size <n|150>                             # Size of chunks from which to extract iVectors"
  echo "  --period <n|50>                                  # Frequency that iVectors are computed"
  echo "  --min-chunks-size <n|25>                         # Minimum chunk-size after splitting larger segments"
  echo "  --use-vad <bool|false>                           # If true, use vad.scp instead of segments"
  exit 1;
fi

srcdir=$1
data=$2
dir=$3

nnet=$srcdir/$iter.raw

for f in $nnet $data/feats.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.

mkdir -p $dir/log
sdata=$data/split${nj}utt;
utils/split_data.sh --per-utt $data $nj || exit 1;

echo $nj > $dir/num_jobs

feats="ark,s,cs:splice-feats --left-context=$left_context --right-context=$right_context scp:$sdata/JOB/feats.scp ark:- |"

export PATH=/home/dsnyder/a16/a16/dsnyder/kaldi/kaldi-nnet3-lid/src/xvectorbin:$PATH

if [ $stage -le 0 ]; then
  echo "$0: extracting iVectors"
  $cmd JOB=1:$nj $dir/log/extract_xvectors.JOB.log \
    /home/dsnyder/a16/a16/dsnyder/kaldi/kaldi-nnet3-lid/src/xvectorbin/nnet3-xvector-compute-simple --use-gpu=no --chunk-size=$chunk_size --min-chunk-size=$min_chunk_size "$nnet" "$feats" \
    ark,scp:$dir/xvector.JOB.ark,$dir/xvector.JOB.scp || exit 1
fi

if [ $stage -le 1 ]; then
  echo "$0: combining xVectors across jobs"
  for j in $(seq $nj); do cat $dir/xvector.$j.scp; done >$dir/xvector.scp || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: Computing mean of iVectors"
  $cmd $dir/log/mean.log \
    ivector-mean scp:$dir/xvector.scp $dir/mean.vec || exit 1;
fi

if [ -z "$pca_dim" ]; then
  pca_dim=`head -n 1 $dir/xvector.scp | copy-vector scp:- ark,t:- | feat-to-dim ark,t:- -` || exit 1
fi

if [ $stage -le 3 ]; then
  echo "$0: Computing whitening transform"
  $cmd $dir/log/transform.log \
    est-pca --read-vectors=true --normalize-mean=false \
      --normalize-variance=true --dim=$pca_dim \
      scp:$dir/xvector.scp $dir/transform.mat || exit 1;
fi

utils/filter_scp.pl $dir/xvector.scp $data/utt2spk > $dir/utt2spk
utils/filter_scp.pl $dir/utt2spk $data/segments > $dir/segments
utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

