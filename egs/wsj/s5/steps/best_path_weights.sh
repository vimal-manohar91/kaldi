#!/bin/bash

# Copyright 2014-17 Vimal Manohar

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# This script combines frame-level posteriors from different decode 
# directories. The first decode directory is assumed to be the primary 
# and is used to get the best path. The posteriors from other decode
# directories are interpolated with the posteriors of the best path. 
# The output is a new directory with final.mdl, tree from the primary 
# decode-dir and the best path alignments and weights in a decode-directory 
# with the same basename as the primary directory.
# This is typically used to get better posteriors for semisupervised training
# of DNN
# e.g. steps/best_path_weights.sh exp/tri6_nnet/decode_train_unt.seg 
# exp/sgmm_mmi_b0.1/decode_fmllr_train_unt.seg_it4 exp/combine_dnn_sgmm
# Here the final.mdl and tree are copied from exp/tri6_nnet to 
# exp/combine_dnn_sgmm. ali.*.gz obtained from the primary dir and 
# the interpolated posteriors in weights.scp are placed in
# exp/combine_dnn_sgmm/decode_train_unt.seg

set -e

# begin configuration section.
cmd=run.pl
stage=-10
acwt=0.1
write_words=false   # Dump the word-level transcript in addition to the best path alignments
#end configuration section.

cat <<EOF
  Usage: $0 [options] <data-dir> <graph-dir|lang-dir> <decode-dir1>[:weight] <decode-dir2>[:weight] [<decode-dir3>[:weight] ... ] <out-dir>
    E.g. $0 data/train_unt.seg data/lang exp/tri1/decode:0.5 exp/tri2/decode:0.25 exp/tri3/decode:0.25 exp/combine
  Options:
    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes.
EOF

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 4 ]; then
  printf "$help_message\n";
  exit 1;
fi

data=$1
lang=$2
dir=${@: -1}  # last argument to the script
shift 2;
decode_dirs=( $@ )  # read the remaining arguments into an array
unset decode_dirs[${#decode_dirs[@]}-1]  # 'pop' the last argument which is odir
num_sys=${#decode_dirs[@]}  # number of systems to combine

mkdir -p $dir
mkdir -p $dir/log

decode_dir=`echo ${decode_dirs[0]} | cut -d: -f1`
nj=`cat $decode_dir/num_jobs`

mkdir -p $dir

words_wspecifier=ark:/dev/null
if $write_words; then
  words_wspecifier="ark,t:| utils/int2sym.pl -f 2- $lang/words.txt > $dir/text.JOB"
fi

if [ $stage -lt -1 ]; then
  mkdir -p $dir/log
  $cmd JOB=1:$nj $dir/log/best_path.JOB.log \
    lattice-best-path --acoustic-scale=$acwt \
      "ark,s,cs:gunzip -c $decode_dir/lat.JOB.gz |" \
      "$words_wspecifier" "ark:| gzip -c > $dir/ali.JOB.gz" || exit 1
fi

if [ -f `dirname $decode_dir`/final.mdl ]; then
  src_dir=`dirname $decode_dir`
else
  src_dir=$decode_dir
fi

cp $src_dir/cmvn_opts $dir/ || exit 1
for f in final.mat splice_opts frame_subsampling_factor; do
  [ -f $src_dir/$f ] && cp $src_dir/$f $dir 
done

weights_sum=0.0

for i in `seq 0 $[num_sys-1]`; do
  decode_dir=${decode_dirs[$i]}

  weight=`echo $decode_dir | cut -d: -s -f2`
  [ -z "$weight" ] && weight=1.0

  if [ $i -eq 0 ]; then
    file_list="\"ark:vector-scale --scale=$weight ark:$dir/weights.$i.JOB.ark ark:- |\""
  else
    file_list="$file_list \"ark,s,cs:vector-scale --scale=$weight ark:$dir/weights.$i.JOB.ark ark:- |\""
  fi

  weights_sum=`perl -e "print STDOUT $weights_sum + $weight"`
done

inv_weights_sum=`perl -e "print STDOUT 1.0/$weights_sum"`

fdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

for i in `seq 0 $[num_sys-1]`; do
  if [ $stage -lt $i ]; then
    decode_dir=`echo ${decode_dirs[$i]} | cut -d: -f1`
    if [ -f `dirname $decode_dir`/final.mdl ]; then
      # model one level up from decode dir
      this_srcdir=`dirname $decode_dir`
    else
      this_srcdir=$decode_dir
    fi

    model=$this_srcdir/final.mdl
    tree=$this_srcdir/tree

    for f in $model $decode_dir/lat.1.gz $tree; do
      [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
    done
    if [ $i -eq 0 ]; then
      nj=`cat $decode_dir/num_jobs` || exit 1;
      cp $model $dir || exit 1
      cp $tree $dir || exit 1
      echo $nj > $dir/num_jobs
    else
      if [ $nj != `cat $decode_dir/num_jobs` ]; then
        echo "$0: number of decoding jobs mismatches, $nj versus `cat $decode_dir/num_jobs`" 
        exit 1;
      fi
    fi

    $cmd JOB=1:$nj $dir/log/get_post.$i.JOB.log \
      lattice-to-post --acoustic-scale=$acwt \
        "ark,s,cs:gunzip -c $decode_dir/lat.JOB.gz|" ark:- \| \
      post-to-pdf-post $model ark,s,cs:- ark:- \| \
      get-post-on-ali ark,s,cs:- "ark,s,cs:gunzip -c $dir/ali.JOB.gz | convert-ali $dir/final.mdl $model $tree ark,s,cs:- ark:- | ali-to-pdf $model ark,s,cs:- ark:- |" "ark,scp:$fdir/weights.$i.JOB.ark,$fdir/weights.$i.JOB.scp" || exit 1
  fi
done

if [ $stage -lt $num_sys ]; then
  if [ "$num_sys" -eq 1 ]; then
    for n in `seq $nj`; do
      cat $dir/weights.0.$n.scp 
    done > $dir/weights.scp
  else
    $cmd JOB=1:$nj $dir/log/interpolate_post.JOB.log \
      vector-sum $file_list ark:- \| \
      vector-scale --scale=$inv_weights_sum ark:- \
      ark,scp:$fdir/weights.JOB.ark,$fdir/weights.JOB.scp || exit 1

    for n in `seq $nj`; do
      cat $dir/weights.$n.scp 
    done > $dir/weights.scp
  fi
fi

for n in `seq 1 $[num_sys-1]`; do
  rm $dir/weights.$n.*.ark $dir/weights.$n.*.scp
done

if $write_words; then
  for n in `seq $nj`; do
    cat $dir/text.$n
  done > $dir/text
fi

exit 0
