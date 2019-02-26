#!/bin/bash

# Copyright 2017     Pegah Ghahremani
#           2017-18  Vimal Manohar
# Apache 2.0

# This script generates examples for multilingual training of 'chain' 
# models using separate input egs dir per language as input.
# This script is similar to steps/nnet3/multilingual/combine_egs.sh, but 
# works on 'chain' egs. This is also useful for semi-supervised training,
# where supervised and unsupervised datasets are treated as different 
# languages.

# This scripts produces 3 sets of files --
# cegs.*.scp, cegs.output.*.ark, cegs.weight.*.ark
#
# cegs.*.scp are the SCP files of the training examples.
# cegs.weight.*.ark map from the key of the example to the language-specific
# weight of that example.
# cegs.output.*.ark map from the key of the example to the name of
# the output-node in the neural net for that specific language, e.g.
# 'output-2'.
#
# Begin configuration section.
cmd=run.pl
block_size=256          # This is the number of consecutive egs that we take from
                        # each source, and it only affects the locality of disk
                        # access.
lang2weight=            # array of weights one per input languge to scale example's output
                        # w.r.t its input language during training.
lang2num_copies=        # comma-separated list of number of copies per 
                        # input language 
                        # This is another way to scale the effect of 
                        # a langauge especially when the language has 
                        # relatively very little data.
affixes=
stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh || exit 1;

set -u -e -o pipefail

if [ $# -lt 3 ]; then
  cat <<EOF
  This script generates examples for multilingual training of neural network
  using separate input egs dir per language as input.
  See top of the script for details.

  Usage: $0 [opts] <num-input-langs,N> <lang1-egs-dir> ...<langN-egs-dir> <multilingual-egs-dir>
   e.g.: $0 [opts] 2 exp/lang1/egs exp/lang2/egs exp/multi/egs

  Options:
      --cmd (utils/run.pl|utils/queue.pl <queue opts>)  # how to run jobs.
      --block-size <int|512>      # it is the number of consecutive egs that we take from 
                                  # each source, and it only affects the locality of disk 
                                  # access. This does not have to be the actual minibatch size
EOF
  exit 1;
fi

num_langs=$1

shift 1
args=("$@")
megs_dir=${args[-1]} # multilingual directory
mkdir -p $megs_dir
mkdir -p $megs_dir/info
if [ ${#args[@]} != $[$num_langs+1] ]; then
  echo "$0: num of input example dirs provided is not compatible with num_langs $num_langs."
  echo "Usage:$0 [opts] <num-input-langs,N> <lang1-egs-dir> ...<langN-egs-dir> <multilingual-egs-dir>"
  echo "Usage:$0 [opts] 2 exp/lang1/egs exp/lang2/egs exp/multi/egs"
  exit 1;
fi

num_copies_per_lang=
if [ ! -z "$lang2num_copies" ]; then
  IFS=, read -r -a num_copies_per_lang <<< $lang2num_copies
  if [ ${#num_copies_per_lang[@]} -ne $num_langs ]; then
    echo "$0: --lang2num-copies must be an array of num-langs=$num_langs integers"
    exit 1
  fi
fi

required="cegs.scp combine.scp train_diagnostic.scp valid_diagnostic.scp"
train_scp_list=
train_diagnostic_scp_list=
valid_diagnostic_scp_list=
combine_scp_list=

# read paramter from $egs_dir[0]/info and cmvn_opts
# to write in multilingual egs_dir.
check_params="info/feat_dim info/ivector_dim info/left_context info/right_context info/left_context_initial info/right_context_final cmvn_opts"
ivec_dim=`cat ${args[0]}/info/ivector_dim`
if [ $ivec_dim -ne 0 ];then check_params="$check_params info/final.ie.id"; fi

for param in $check_params info/frames_per_eg; do
  cat ${args[0]}/$param > $megs_dir/$param || exit 1;
done

rm -f $megs_dir/info/graph_info

if [ -f ${args[0]}/0.trans_mdl ]; then
  cp ${args[0]}/{0.trans_mdl,tree,den.fst,normalization.fst} $megs_dir
fi

if [ -z "$lang2num_copies" ]; then
  num_copies_per_lang=($(for n in $(seq $num_langs); do echo -n 1" "; done))
fi

declare -a affixes_per_lang

if [ -z "$affixes" ]; then
  for n in $(seq 0 $[num_langs-1]); do
    affixes_per_lang[$n]=
  done
else
  affixes_per_lang=($affixes)
fi

tot_num_archives=0
for lang in $(seq 0 $[$num_langs-1]); do
  this_egs_dir=${args[$lang]}
  for f in $required; do
    if [ ! -f $this_egs_dir/$f ]; then
      echo "$0: no such file $this_egs_dir/$f." && exit 1;
    fi
  done

  if { [ -z "$lang2num_copies" ] || [ ${num_copies_per_lang[$lang]} -eq 1 ]; } && [ -z "${affixes_per_lang[$lang]}" ]; then
    train_scp_list="$train_scp_list $this_egs_dir/cegs.scp"
    train_diagnostic_scp_list="$train_diagnostic_scp_list $this_egs_dir/train_diagnostic.scp"
    valid_diagnostic_scp_list="$valid_diagnostic_scp_list $this_egs_dir/valid_diagnostic.scp"
    combine_scp_list="$combine_scp_list $this_egs_dir/combine.scp"
    num_archives=$(cat $this_egs_dir/info/num_archives)
  else
    rm -f $megs_dir/lang${lang}_cegs.scp $megs_dir/lang${lang}_train_diagnostic.scp \
      $megs_dir/lang${lang}_valid_diagnostic.scp $megs_dir/lang${lang}_combine.scp

    if [ $(perl -e "{print int(${num_copies_per_lang[$lang]})}") != ${num_copies_per_lang[$lang]} ]; then
      echo "$0: Expected --lang2num-copies to have only integers; "
      echo "$0: got ${num_copies_per_lang[$lang]} for language $lang"
      exit 1
    fi

    for i in `seq ${num_copies_per_lang[$lang]}`; do
      awk -v i=$i -v a=${affixes_per_lang[$lang]} '{print $1 a"-"i" "$2}' $this_egs_dir/cegs.scp >> \
        $megs_dir/lang${lang}_cegs.scp
      awk -v i=$i -v a=${affixes_per_lang[$lang]} '{print $1 a"-"i" "$2}' $this_egs_dir/train_diagnostic.scp >> \
        $megs_dir/lang${lang}_train_diagnostic.scp
      awk -v i=$i -v a=${affixes_per_lang[$lang]} '{print $1 a"-"i" "$2}' $this_egs_dir/valid_diagnostic.scp >> \
        $megs_dir/lang${lang}_valid_diagnostic.scp
      awk -v i=$i -v a=${affixes_per_lang[$lang]} '{print $1 a"-"i" "$2}' $this_egs_dir/combine.scp >> \
        $megs_dir/lang${lang}_combine.scp
    done

    if [ $(head -n1 $megs_dir/lang${lang}_cegs.scp | wc -w) -ne 2 ]; then
      echo "$0: Incorrect format in $megs_dir/lang${lang}_cegs.scp; something went wrong!"
      exit 1
    fi

    train_scp_list="$train_scp_list $megs_dir/lang${lang}_cegs.scp"
    train_diagnostic_scp_list="$train_diagnostic_scp_list $megs_dir/lang${lang}_train_diagnostic.scp"
    valid_diagnostic_scp_list="$valid_diagnostic_scp_list $megs_dir/lang${lang}_valid_diagnostic.scp"
    combine_scp_list="$combine_scp_list $megs_dir/lang${lang}_combine.scp"

    num_archives=$(cat $this_egs_dir/info/num_archives)
    num_archives=$[num_archives * ${num_copies_per_lang[$lang]}]
  fi
  tot_num_archives=$[tot_num_archives+num_archives]

  # check parameter dimension to be the same in all egs dirs
  for f in $check_params; do
    if [ -f $megs_dir/$f ] && [ -f $this_egs_dir/$f ]; then
      f1=$(cat $megs_dir/$f)
      f2=$(cat $this_egs_dir/$f)
      if [ "$f1" != "$f2" ]  ; then
        echo "$0: mismatch for $f in $megs_dir vs. $this_egs_dir($f1 vs. $f2)."
        exit 1;
      fi
    else
      echo "$0: file $f does not exits in $megs_dir or $this_egs_dir/$f ."
    fi
  done

  if [ -f ${args[$lang]}/den.fst ]; then
    for f in 0.trans_mdl tree den.fst normalization.fst; do
      if [ ! -f $this_egs_dir/$f ]; then
        echo "$0: Could not find $this_egs_dir/$f"
        exit 1
      fi
    done

    this_egs_dir=$(utils/make_absolute.sh $this_egs_dir)
    echo "output-$lang $this_egs_dir/0.trans_mdl $this_egs_dir/tree $this_egs_dir/den.fst $this_egs_dir/normalization.fst" >> $megs_dir/info/graph_info
  fi
done

if [ ! -z "$lang2weight" ]; then
  egs_opt="--lang2weight '$lang2weight'"
fi

if [ $stage -le 0 ]; then
  echo "$0: allocating multilingual examples for training."
  # Generate cegs.*.scp for multilingual setup.
  $cmd $megs_dir/log/allocate_multilingual_examples_train.log \
    steps/nnet3/multilingual/allocate_multilingual_examples.py $egs_opt \
      --num-archives $tot_num_archives \
      --block-size $block_size \
      --egs-prefix "cegs." \
      $train_scp_list $megs_dir || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combine combine.scp examples from all langs in $megs_dir/combine.scp."
  # Generate combine.scp for multilingual setup.
  $cmd $megs_dir/log/allocate_multilingual_examples_combine.log \
    steps/nnet3/multilingual/allocate_multilingual_examples.py $egs_opt \
      --num-archives 1 \
      --block-size $block_size \
      --egs-prefix "combine." \
      $combine_scp_list $megs_dir || exit 1;

  echo "$0: combine train_diagnostic.scp examples from all langs in $megs_dir/train_diagnostic.scp."
  # Generate train_diagnostic.scp for multilingual setup.
  $cmd $megs_dir/log/allocate_multilingual_examples_train_diagnostic.log \
    steps/nnet3/multilingual/allocate_multilingual_examples.py $egs_opt \
      --num-archives 1 \
      --block-size $block_size \
      --egs-prefix "train_diagnostic." \
      $train_diagnostic_scp_list $megs_dir || exit 1;


  echo "$0: combine valid_diagnostic.scp examples from all langs in $megs_dir/valid_diagnostic.scp."
  # Generate valid_diagnostic.scp for multilingual setup.
  $cmd $megs_dir/log/allocate_multilingual_examples_valid_diagnostic.log \
    steps/nnet3/multilingual/allocate_multilingual_examples.py $egs_opt \
      --num-archives 1 \
      --block-size $block_size \
      --egs-prefix "valid_diagnostic." \
      $valid_diagnostic_scp_list $megs_dir || exit 1;

fi
for egs_type in combine train_diagnostic valid_diagnostic; do
  mv $megs_dir/${egs_type}.output.1.ark $megs_dir/${egs_type}.output.ark || exit 1;
  mv $megs_dir/${egs_type}.weight.1.ark $megs_dir/${egs_type}.weight.ark || exit 1;
  mv $megs_dir/${egs_type}.1.scp $megs_dir/${egs_type}.scp || exit 1;
done
echo $tot_num_archives > $megs_dir/info/num_archives || exit 1;
echo $num_langs > $megs_dir/info/num_tasks || exit 1;
echo "$0: Finished preparing multilingual training example."
