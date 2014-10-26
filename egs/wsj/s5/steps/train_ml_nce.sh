#!/bin/bash
# Copyright 2014  Vimal Manohar (Johns Hopkins University)

# Semisupervised discriminative training using ML objective for 
# supervised data and NCE objective for unsupervised data
# 4 iterations (by default) of Extended Baum-Welch update.
#
# For the numerator we have a fixed alignment rather than a lattice--
# this actually follows from the way lattices are defined in Kaldi, which
# is to have a single path for each word (output-symbol) sequence.

# Begin configuration section.
cmd=run.pl
num_iters=4
tau=400
weight_tau=10
alpha=0.1
acwt=0.1
transform_dir_unsup=
stage=0
# End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 6 ]; then
  echo "Usage: steps/train_ml_nce.sh <data-sup> <data-unsup> <lang> <ali> <lats> <exp>"
  echo " e.g.: steps/train_ml_nce.sh data/train data/unsup data/lang exp/tri3b_ali exp/tri3b/decode_unsup exp/tri3b_ml_nce"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  echo "  --tau                                            # tau for i-smooth to last iter (default 200)"
  echo "  --alpha                                          # scale unsupervised stats relative to supervised stats"
  
  exit 1;
fi

data_sup=$1
data_unsup=$2
lang=$3
alidir=$4
latdir=$5
dir=$6

mkdir -p $dir/log

for f in $data_unsup/feats.scp $data_sup/feats.scp $alidir/{tree,final.mdl,ali.1.gz} $latdir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
nj_sup=`cat $alidir/num_jobs` || exit 1;
nj_unsup=`cat $latdir/num_jobs` || exit 1;

sdata_sup=$data_sup/split$nj_sup
sdata_unsup=$data_unsup/split$nj_unsup
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
mkdir -p $dir/log
cp $alidir/splice_opts $dir 2>/dev/null
cp $alidir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
[[ -d $sdata_sup && $data_sup/feats.scp -ot $sdata_sup ]] || split_data.sh $data_sup $nj_sup || exit 1;
[[ -d $sdata_unsup && $data_unsup/feats.scp -ot $sdata_unsup ]] || split_data.sh $data_unsup $nj_unsup || exit 1;
echo $nj_sup > $dir/num_jobs_sup
echo $nj_unsup > $dir/num_jobs_unsup

cp $alidir/tree $dir
cp $alidir/final.mdl $dir/0.mdl

silphonelist=`cat $lang/phones/silence.csl` || exit 1;

# Set up features

if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats_sup="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata_sup/JOB/utt2spk scp:$sdata_sup/JOB/cmvn.scp scp:$sdata_sup/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats_sup="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata_sup/JOB/utt2spk scp:$sdata_sup/JOB/cmvn.scp scp:$sdata_sup/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

case $feat_type in
  delta) feats_unsup="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata_unsup/JOB/utt2spk scp:$sdata_unsup/JOB/cmvn.scp scp:$sdata_unsup/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats_unsup="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata_unsup/JOB/utt2spk scp:$sdata_unsup/JOB/cmvn.scp scp:$sdata_unsup/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

[ -f $alidir/trans.1 ] && echo Using transforms from $alidir && \
  feats_sup="$feats_sup transform-feats --utt2spk=ark:$sdata_sup/JOB/utt2spk ark,s,cs:$alidir/trans.JOB ark:- ark:- |"

[ -z "$transform_dir_unsup" ] && echo "$0: --transform-dir was not specified. Trying $latdir as transform_dir" && transform_dir_unsup=$latdir

[ -f $transform_dir_unsup/trans.1 ] && echo Using transforms from $transform_dir_unsup && \
  feats_unsup="$feats_unsup transform-feats --utt2spk=ark:$sdata_unsup/JOB/utt2spk ark,s,cs:$transform_dir_unsup/trans.JOB ark:- ark:- |"

lats="ark:gunzip -c $latdir/lat.JOB.gz|"

cur_mdl=$alidir/final.mdl

x=0
while [ $x -lt $num_iters ]; do
  echo "Iteration $x of ML-NCE training"
  # Note: the num and den states are accumulated at the same time, so we
  # can cancel them per frame.
  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj_sup $dir/log/acc.$x.JOB.log \
      gmm-acc-stats $dir/$x.mdl "$feats_sup" "ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-post ark:- ark:- |" \
      $dir/num_acc.$x.JOB.acc || exit 1;

    n=`echo $dir/num_acc.$x.*.acc | wc -w`;
    [ "$n" -ne $[$nj_sup] ] && \
      echo "Wrong number of ML accumulators $n versus $nj_sup" && exit 1;
    $cmd $dir/log/num_acc_sum.$x.log \
      gmm-sum-accs $dir/num_acc.$x.acc $dir/num_acc.$x.*.acc || exit 1;
    rm $dir/num_acc.$x.*.acc
    
    $cmd JOB=1:$nj_unsup $dir/log/lat_acc.$x.JOB.log \
      gmm-rescore-lattice $cur_mdl "$lats" "$feats_unsup" ark:- \| \
      lattice-to-nce-post --acoustic-scale=$acwt $cur_mdl \
        ark:- ark:- \| \
      gmm-acc-stats $cur_mdl "$feats_unsup" ark,s,cs:- \
        $dir/lat_acc.$x.JOB.acc || exit 1;
    
    n=`echo $dir/lat_acc.$x.*.acc | wc -w`;
    [ "$n" -ne $[$nj_unsup] ] && \
      echo "Wrong number of NCE accumulators $n versus $nj_unsup" && exit 1;
    $cmd $dir/log/lat_acc_sum.$x.log \
      gmm-sum-accs $dir/lat_acc.$x.acc $dir/lat_acc.$x.*.acc || exit 1;
    rm $dir/lat_acc.$x.*.acc

  # note: this tau value is for smoothing towards model parameters, not
  # as in the Boosted MMI paper, not towards the ML stats as in the earlier
  # work on discriminative training (e.g. my thesis).  
  # You could use gmm-ismooth-stats to smooth to the ML stats, if you had
  # them available [here they're not available if cancel=true].

    $cmd $dir/log/update.$x.log \
      gmm-est-gaussians-ebw --tau=$tau $dir/$x.mdl \
      "gmm-sum-accs - $dir/num_acc.$x.acc \"gmm-scale-accs $alpha $dir/lat_acc.$x.acc - |\" |" \
      "gmm-scale-accs 0.0 $dir/num_acc.$x.acc - |" - \| \
      gmm-est-weights-ebw --weight-tau=$weight_tau - \
      "gmm-sum-accs - $dir/num_acc.$x.acc \"gmm-scale-accs $alpha $dir/lat_acc.$x.acc - |\" |" \
      "gmm-scale-accs 0.0 $dir/num_acc.$x.acc - |" $dir/$[$x+1].mdl || exit 1;
    rm $dir/{num,lat}_acc.$x.acc
  fi

  # Some diagnostics: the objective function progress and auxiliary-function
  # improvement.

  tail -n 50 $dir/log/acc.$x.*.log | perl -e '$acwt=shift @ARGV; while(<STDIN>) { if(m/gmm-acc-stats.+Overall weighted acoustic likelihood per frame was (\S+) over (\S+) frames/) { $tot_aclike += $1*$2; $tot_frames1 += $2; } } $tot_aclike *= ($acwt / $tot_frames1);  $num_like = $tot_aclike; $per_frame_objf = $num_like; print "$per_frame_objf $tot_frames1\n"; ' $acwt > $dir/tmpf
  objf_ml=`cat $dir/tmpf | awk '{print $1}'`;
  nf_ml=`cat $dir/tmpf | awk '{print $2}'`;
  rm $dir/tmpf

 tail -n 50 $dir/log/acc.$x.*.log | perl -e 'while(<STDIN>) { if(m/lattice-to-nce-post.+Overall average Negative Conditional Entropy is (\S+) over (\S+) frames/) { $tot_objf += $1*$2; $tot_frames += $2; }} $tot_objf /= $tot_frames; print "$tot_objf*$ARGV[0] $tot_frames\n"; ' $alpha> $dir/tmpf
  objf_nce=`cat $dir/tmpf | awk '{print $1}'`;
  nf_nce=`cat $dir/tmpf | awk '{print $2}'`;
  rm $dir/tmpf

  nf=`perl -e "print $nf_nce + $nf_ml"`
  impr=`grep -w Overall $dir/log/update.$x.log | awk '{x += $10*$12;} END{print x;}'`
  impr=`perl -e "print ($impr*$acwt/$nf);"` # We multiply by acwt, and divide by $nf which is the "real" number of frames.
  echo "Iteration $x: ML objf was $objf_ml, NCE objf was $objf_nce, auxf change was $impr" | tee $dir/objf.$x.log

  x=$[$x+1]
done

echo "ML-NCE training finished"

rm $dir/final.mdl 2>/dev/null
ln -s $x.mdl $dir/final.mdl

exit 0;


