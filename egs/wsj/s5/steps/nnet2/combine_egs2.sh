#!/bin/bash

# Copyright 2015 Vimal Manohar (Johns Hopkins University)
# Apache 2.0

# This script is used to combine multiple neural network example directories.
# The training and validation subsets are directly copied from the first 
# example directory.

# The example directories must have been created in the egs2 format using 
# a script like get_egs2.sh

# Begin configuration section.
cmd=run.pl
samples_per_iter=400000 # each iteration of training, see this many samples
                        # per job.  This is just a guideline; it will pick a number
                        # that divides the number of samples in the entire data.
stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 3 ]; then
  echo "Usage: $0 [opts] <data> <dest-egs-dir> <egs-dir1> <egs-dir2> ..."
  echo "Note, the examples for diagnostics and combine are taken only from the first egs directory."
  exit 1
fi

dir=$1
shift;

rm -r $dir 2>/dev/null
mkdir -p $dir;

export LC_ALL=C

num_egs_dirs=$#
num_frames=0

first_src=$1
for f in combine.egs train_diagnostic.egs valid_diagnostic.egs cmvn_opts splice_opts tree final.mat final.mdl; do
  if [ ! -f $first_src/$f ]; then
    echo "$0: no such file $first_src/$f"
    exit 1;
  fi
  cp $first_src/$f $dir
done
frames_per_eg=`cat $first_src/info/frames_per_eg` || exit 1
[ -z "$frames_per_eg" ] && echo "$0: Unable to read $first_src/info/frames_per_eg" && exit 1

mismatch_frames_per_eg=false
for d in $*; do
  for f in info/frames_per_eg info/num_frames info/num_archives egs.1.ark; do
    if [ ! -f $d/$f ]; then
      echo "$0: no such file $d/$f"
      exit 1;
    fi
  done
  if [ `cat $d/info/frames_per_eg` -ne $frames_per_eg ]; then
    echo "$0: mismatch in frames_per_eg; `cat $d/info/frames_per_eg` vs $frames_per_eg"
    echo "$0: Choosing frames_per_eg to be 1"
    mismatch_frames_per_eg=true
    frames_per_eg=1
  fi

  this_num_frames=`cat $d/info/num_frames`
  num_frames=$[num_frames+this_num_frames]
done

mkdir -p $dir/log $dir/info

cp $first_src/info/* $dir/info

num_archives=$[num_frames/(frames_per_eg * samples_per_iter)+1]
echo $num_frames >$dir/info/num_frames
echo $num_archives >$dir/info/num_archives
echo $frames_per_eg >$dir/info/frames_per_eg

# Working out number of egs per archive
egs_per_archive=$[$num_frames/($frames_per_eg*$num_archives)]
! [ $egs_per_archive -le $samples_per_iter ] && \
  echo "$0: script error: egs_per_archive=$egs_per_archive not <= samples_per_iter=$samples_per_iter" \
  && exit 1;

echo $egs_per_archive > $dir/info/egs_per_archive
echo "$0: creating $num_archives archives, each with $egs_per_archive egs, with"
echo "$0:   $frames_per_eg labels per example"

# Making soft links to storage directories.  This is a no-up unless
# the subdirectory $dir/storage/ exists.  See utils/create_split_dir.pl
egs_list=
for x in `seq $num_archives`; do
  utils/create_data_link.pl $dir/egs.$x.ark
  egs_list="$egs_list ark:$dir/egs.$x.ark"
done

if [ $stage -le 1 ]; then
  n=0
  for d in $*; do 
    n=$[n+1]
    if $mismatch_frames_per_eg; then
      this_frames_per_eg=`cat $d/info/frames_per_eg`
      for f in `seq 0 $[this_frames_per_eg-1]`; do
        $cmd $dir/log/copy_egs.$n.$f.log \
          nnet-copy-egs --random=true --frame=$f "ark:cat $d/egs.*.ark |"$egs_list || exit 1
      done
    else
      $cmd $dir/log/copy_egs.$n.log \
        nnet-copy-egs --random=true "ark:cat $d/egs.*.ark |"$egs_list || exit 1
    fi
  done
fi

echo "$0: Finished combining training examples"
