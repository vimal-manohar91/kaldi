#!/bin/bash

# Copyright 2014  Guoguo Chen
#           2017  Vimal Manohar
# Apache 2.0

# This script rescores non-compact, (possibly) undeterminized lattices with the 
# ConstArpaLm format language model.
# This is similar to steps/lmrescore_const_arpa.sh, but expects 
# non-compact lattices as input.
# This works by first determinizing the lattice and rescoring it with 
# const ARPA LM, followed by composing it with the original lattice to add the 
# new LM scores.

# If you use the option "--write compact false" it outputs non-compact lattices;
# the purpose is to add in LM scores while leaving the frame-by-frame acoustic
# scores in the same position that they were in in the input, undeterminized
# lattices. This is important in our 'chain' semi-supervised training recipes,
# where it helps us to split lattices while keeping the scores at the edges of
# the split points correct.

# Begin configuration section.
cmd=run.pl
keep_subsplit=false  # If true, then retain the lattices corresponding to sub-split data.
                     # This is only applicable if reading from a lattice directory that
                     # has lattices corresponding to the sub-splits and sub_split > 1.
skip_scoring=false
stage=1
scoring_opts=
write_compact=true   # If set to false, writes lattice in non-compact format.
                     # This retains the acoustic scores on the arcs of the lattice.
                     # Useful for another stage of LM rescoring.
acwt=0.1  # used for pruning and determinization
beam=8.0  # beam used in determinization

# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./utils/parse_options.sh

if [ $# != 5 ]; then
  cat <<EOF
   Does language model rescoring of non-compact undeterminized lattices 
   (remove old LM, add new LM). This script expects the input lattices 
   to be in non-compact format.
   Usage: $0 [options] <old-lang-dir> <new-lang-dir> \\
                      <data-dir> <input-decode-dir> <output-decode-dir>
   options: [--cmd (run.pl|queue.pl [queue opts])]
   See also: steps/lmrescore_const_arpa.sh 
EOF
   exit 1;
fi

[ -f path.sh ] && . ./path.sh;

oldlang=$1
newlang=$2
data=$3
indir=$4
outdir=$5

oldlm=$oldlang/G.fst
newlm=$newlang/G.carpa
! cmp $oldlang/words.txt $newlang/words.txt &&\
  echo "$0: Warning: vocabularies may be incompatible."
[ ! -f $oldlm ] && echo "$0: Missing file $oldlm" && exit 1;
[ ! -f $newlm ] && echo "$0: Missing file $newlm" && exit 1;
! ls $indir/lat.*.gz >/dev/null &&\
  echo "$0: No lattices input directory $indir" && exit 1;

if ! cmp -s $oldlang/words.txt $newlang/words.txt; then
  echo "$0: $oldlang/words.txt and $newlang/words.txt differ: make sure you know what you are doing.";
fi

oldlmcommand="fstproject --project_output=true $oldlm |"

mkdir -p $outdir/log
nj=$(cat $indir/num_jobs) || exit 1;
cp $indir/num_jobs $outdir

sub_split=1
if [ -f $indir/sub_split ]; then
  sub_split=$(cat $indir/sub_split) || exit 1
fi

if [ $stage -le 1 ]; then
  if [ $sub_split -eq 1 ]; then
    # Normal case. Useful for small size datasets.
    lats_rspecifier="ark:gunzip -c $indir/lat.JOB.gz |"
    lats_wspecifier="ark:| gzip -c > $outdir/lat.JOB.gz"

    # 1. Determinize the lattice since the input lattice is not determinized
    #    and the rescoring binary expects determinized lattices.
    # 2. Remove the costs from the determinized lattice.
    # 3. Add negated LM cost from the old LM. This will remove the
    #    old LM costs from the original lattice when composed with it.
    #    Note that we add only the costs from the best path in the old LM. 
    #    This is done by determinizing the lattice at the end of this step.
    #    Since we first determinized the original lattice, determinizing the
    #    lattice here is equivalent to taking the best path in the old LM.
    # 4. Add the LM costs from the new LM. This will add the new LM costs
    #    to the original lattice when composed with it.
    #    Note that we add only the costs from the best path in the new LM.
    #    This is done by determinizing the lattice at the end of this step.
    #    Since we first determinized the original lattice, determinizing the
    #    lattice here is equivalent to taking the best path in the new LM.
    # 5. Project the lattice the olabels. This is probably not necessary.
    # 6. Compose the rescored determinized lattice with the original lattice.
    #    The only acoustic costs come from the original lattice. 
    #    The LM costs in the output lattice is LM costs in the original lattice 
    #    - costs of the old LM + costs of the new LM.

    $cmd JOB=1:$nj $outdir/log/rescorelm.JOB.log \
      lattice-determinize-pruned --acoustic-scale=$acwt --beam=$beam \
        "ark:gunzip -c $indir/lat.JOB.gz |" ark:- \| \
      lattice-scale --lm-scale=0.0 --acoustic-scale=0.0 ark:- ark:- \| \
      lattice-lmrescore --lm-scale=-1.0 ark:- "$oldlmcommand" ark:- \| \
      lattice-lmrescore-const-arpa --lm-scale=1.0 \
        ark:- "$newlm" ark:- \| \
      lattice-project ark:- ark:- \| \
      lattice-compose --write-compact=$write_compact \
        "$lats_rspecifier" \
        ark,s,cs:- "$lats_wspecifier" || exit 1
  else
    # each job from 1 to $nj is split into multiple pieces (sub-split), and we aim
    # to have at most two jobs running at each time.  The idea is that if we have
    # stragglers from one job, we can be processing another one at the same time.
    rm $dir/.error 2>/dev/null

    prev_pid=
    for n in $(seq $[nj+1]); do
      lats_rspecifier="ark:gunzip -c $indir/lat.$n.JOB.gz |"
      lats_wspecifier="ark:| gzip -c > $outdir/lat.$n.JOB.gz"

      if [ $n -gt $nj ]; then
        this_pid=
      elif [ -f $dir/.done.$n ] && [ $dir/.done.$n -nt $model ]; then
        echo "$0: Not processing subset $n as already done (delete $dir/.done.$n if not)";
        this_pid=
      else
        mkdir -p $dir/log/$n
        mkdir -p $dir/part

        $cmd JOB=1:$nj $outdir/log/rescorelm.JOB.log \
          lattice-determinize-pruned --acoustic-scale=$acwt --beam=$beam \
            "ark:gunzip -c $indir/lat.JOB.gz |" ark:- \| \
          lattice-scale --lm-scale=0.0 --acoustic-scale=0.0 ark:- ark:- \| \
          lattice-lmrescore --lm-scale=-1.0 ark:- "$oldlmcommand" ark:- \| \
          lattice-lmrescore-const-arpa --lm-scale=1.0 \
            ark:- "$newlm" ark:- \| \
          lattice-project ark:- ark:- \| \
          lattice-compose --write-compact=$write_compact \
            "$lats_rspecifier" \
            ark,s,cs:- "$lats_wspecifier" || touch $dir/.error &
        this_pid=$!
      fi
      if [ ! -z "$prev_pid" ]; then # Wait for the previous job to merge lattices.
        wait $prev_pid
        [ -f $dir/.error ] && \
          echo "$0: error generating lattices" && exit 1;

        if ! $keep_subsplit; then
          # If we are not keeping the subsplits, then merge the lattices and 
          # remove the subsplits.
          rm $dir/.merge_error 2>/dev/null
          echo "$0: Merging archives for data subset $prev_n"
          for k in $(seq $sub_split); do
            gunzip -c $dir/lat.$prev_n.$k.gz || touch $dir/.merge_error;
          done | gzip -c > $dir/lat.$prev_n.gz || touch $dir/.merge_error;
          [ -f $dir/.merge_error ] && \
            echo "$0: Merging lattices for subset $prev_n failed" && exit 1;
          rm $dir/lat.$prev_n.*.gz
        fi
        touch $dir/.done.$prev_n
      fi
      prev_n=$n
      prev_pid=$this_pid
    done
  fi
fi

if $keep_subsplit; then
  echo $sub_split > $dir/sub_split
fi

if ! $skip_scoring && [ $stage -le 2 ]; then
  err_msg="Not scoring because local/score.sh does not exist or not executable."
  [ ! -x local/score.sh ] && echo $err_msg && exit 1;
  local/score.sh --cmd "$cmd" $scoring_opts $data $newlang $outdir
else
  echo "Not scoring because requested so..."
fi

exit 0;
