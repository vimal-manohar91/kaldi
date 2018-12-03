#!/bin/bash

cmd=run.pl
cleanup=false
stage=0
iter=final
scoring_opts=
skip_diagnostics=false
skip_scoring=false

. ./path.sh
. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
  exit 1
fi

graphdir=$1
data=$2
dir=$3

nj=$(cat $dir/num_jobs) || exit 1

sub_split=1
if [ -f $dir/sub_split ]; then
  sub_split=$(cat $dir/sub_split)
fi

if [ $sub_split -eq 1 ]; then
  echo "sub-split == 1. Nothing to do here."
  exit 0
fi

if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/merge_lattices.JOB.log \
    for n in $(seq $sub_split)\; do \
      gunzip -c $dir/lat.JOB.\$n.gz\; \
    done \| gzip -c '>' $dir/lat.JOB.gz || exit 1
fi

if [ $stage -le 2 ]; then
  if ! $skip_diagnostics ; then
    [ ! -z $iter ] && iter_opt="--iter $iter"
    steps/diagnostic/analyze_lats.sh --cmd "$cmd" $iter_opt $graphdir $dir
  fi
fi


# The output of this script is the files "lat.*.gz"-- we'll rescore this at
# different acoustic scales to get the final output.
if [ $stage -le 3 ]; then
  if ! $skip_scoring ; then
    [ ! -x local/score.sh ] && \
      echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
    echo "score best paths"
    [ "$iter" != "final" ] && iter_opt="--iter $iter"
    local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
    echo "score confidence and timing with sclite"
  fi
fi

if $cleanup; then
  for n in $(seq $nj); do
    rm $dir/lat.$n.*.gz
  done
fi

echo "Decoding done."
exit 0;
