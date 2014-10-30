#!/bin/bash
# Copyright 2014  Vimal Manohar

# Unsupervised discriminative training using NCE objective
# 4 iterations (by default) of Extended Baum-Welch update.

# Begin configuration section.
cmd=run.pl
num_iters=4
tau=100
weight_tau=10
alpha=1.0
acwt=0.1
stage=0
transform_dir=
update_flags="mv"
# End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "Usage: steps/train_mmi.sh <data> <lang> <lats-dir> <exp>"
  echo " e.g.: steps/train_mmi.sh data/unsup data/lang exp/tri3b/decode_unsup exp/tri3b"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  echo "  --tau                                            # tau for i-smooth to last iter (default 200)"
  
  exit 1;
fi

data=$1
lang=$2
latdir=$3
dir=$4
mkdir -p $dir/log

srcdir=`dirname $latdir`
for f in $data/feats.scp $srcdir/{tree,final.mdl} $latdir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
nj=`cat $latdir/num_jobs` || exit 1;

sdata=$data/split$nj
splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
mkdir -p $dir/log
cp $srcdir/splice_opts $dir 2>/dev/null
cp $srcdir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

cp $srcdir/{final.mdl,tree} $dir

silphonelist=`cat $lang/phones/silence.csl` || exit 1;

# Set up features

if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    cp $srcdir/final.mat $dir    
    ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

if [ ! -z "$transform_dir" ] && [ ! -f $transform_dir/trans.1 ]; then
  [ ! -f $transform_dir/raw_trans.1 ] && echo "transform_dir specified as $transform_dir; but $transform_dir/trans.1 or $transform_dir/raw_trans.1 not found" && exit 1
fi

[ -z "$transform_dir" ] && echo "$0: --transform-dir was not specified. Trying $latdir as transform_dir" && transform_dir=$latdir

[ -f $transform_dir/trans.1 ] && echo Using transforms from $transform_dir && \
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$transform_dir/trans.JOB ark:- ark:- |"

lats="ark:gunzip -c $latdir/lat.JOB.gz|"


cur_mdl=$srcdir/final.mdl
x=0
while [ $x -lt $num_iters ]; do
  echo "Iteration $x of NCE training"
  # Note: the num and den states are accumulated at the same time, so we
  # can cancel them per frame.
  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-rescore-lattice $cur_mdl "$lats" "$feats" ark:- \| \
      lattice-to-nce-post --acoustic-scale=$acwt $cur_mdl \
        ark:- ark:- \| \
      gmm-acc-stats $cur_mdl "$feats" ark,s,cs:- \
        $dir/$x.JOB.acc || exit 1;

    n=`echo $dir/$x.*.acc | wc -w`;
    $cmd $dir/log/acc_sum.$x.log \
      gmm-sum-accs $dir/$x.acc $dir/$x.*.acc || exit 1;
    rm $dir/$x.*.acc
    
    $cmd $dir/log/update.$x.log \
      gmm-est-gaussians-ebw --tau=$tau --update-flags=$update_flags \
      $cur_mdl \
      "gmm-scale-accs 0.0 $dir/$x.acc - |" \
      "gmm-scale-accs -$alpha $dir/$x.acc - |" - \| \
      gmm-est-weights-ebw --weight-tau=$weight_tau \
      --update-flags=$update_flags - \
      "gmm-scale-accs 0.0 $dir/$x.acc - |" \
      "gmm-scale-accs -$alpha $dir/$x.acc - |" $dir/$[$x+1].mdl || exit 1;
    rm $dir/$x.acc
  fi
  cur_mdl=$dir/$[$x+1].mdl

  # Some diagnostics: the objective function progress and auxiliary-function
  # improvement.

 tail -n 50 $dir/log/acc.$x.*.log | perl -e 'while(<STDIN>) { if(m/lattice-to-nce-post.+Overall average Negative Conditional Entropy is (\S+) over (\S+) frames/) { $tot_objf += $1*$2; $tot_frames += $2; }} $tot_objf /= $tot_frames; print "$tot_objf $tot_frames\n"; ' > $dir/tmpf
  objf=`cat $dir/tmpf | awk '{print $1}'`;
  nf=`cat $dir/tmpf | awk '{print $2}'`;
  rm $dir/tmpf
  impr=`grep -w Overall $dir/log/update.$x.log | awk '{x += $10*$12;} END{print x;}'`
  impr=`perl -e "print ($impr*$acwt/$nf);"` # We multiply by acwt, and divide by $nf which is the "real" number of frames.
  # This gives us a projected objective function improvement.
  echo "Iteration $x: objf was $objf, NCE auxf change was $impr" | tee $dir/objf.$x.log
  x=$[$x+1]
done

echo "NCE training finished"

rm $dir/final.mdl 2>/dev/null
ln -s $x.mdl $dir/final.mdl

exit 0;

