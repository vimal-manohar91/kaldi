#!/bin/bash

# Copyright 2014  Vimal Manohar
# Apache 2.0.

set -e 
set -u
set -o pipefail 

# This script does supervised MMI + unsupervised NCE training.
# Note: the temporary data is put in <exp-dir>/uegs/, <exp-dir>/degs/ so if you
# want to use a different disk for that, just make that a soft link to some
# other volume.

# Begin configuration section.
cmd=run.pl
num_epochs=4       # Number of epochs of training
learning_rate=9e-5  # Currently we support same learning rate for supervised
                    # and unsupervised. You can additionally add a scale on
                    # the unsupervised examples if needed.
acoustic_scale=0.1  # acoustic scale
criterion=smbr
boost=0.0         # option relevant for MMI
drop_frames=false #  option relevant for MMI
unsupervised_scale=1.0  # Add a scaling on the unsupervised objective function
num_jobs_nnet=4    # Number of neural net jobs to run in parallel.  Note: this
                   # will interact with the learning rates (if you decrease
                   # this, you'll have to decrease the learning rate, and vice
                   # versa).
samples_per_iter=400000 # measured in frames, not in "examples". This is the
                        # maximum including both supervised and unsupervised
                        # frames
modify_learning_rates=false
last_layer_factor=1.0  # relates to modify-learning-rates
first_layer_factor=1.0 # relates to modify-learning-rates
shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.
stage=-14
max_jobs_run=5  # jobs with a lot of I/O, limits the number running at one time
                # This option replaces io_opts='-tc 5'.
num_threads=16  # this is the default but you may want to change it, e.g. to 1 if
                # using GPUs.
mem=1G          
feat_type=
transform_dir_sup= # If this is a SAT system, directory for transforms
transform_dir_unsup= # If this is a SAT system, directory for transforms
cleanup=false
uegs_dir=
degs_dir=
retroactive=false
online_ivector_dir_unsup=
online_ivector_dir_sup=
use_preconditioning=false
prior_subset_size=10000 # 10k samples per job, for computing priors.  Should be
                        # more than enough.
src_model=
# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 8 ]; then
  echo "Usage: $0 [opts] <data-sup> <data-unsup> <lang> <ali-dir> <den-lat-dir> <lat-dir> <src-dir> <exp-dir>"
  echo " e.g.: $0 data/train data/unsup.uem data/lang exp/tri4_ali
  exp/tri4_denlats exp/tri4_nnet/decode_unsup.uem exp/tri4_nnet
  exp/tri4_nnet_mmi_nce"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-epochs <#epochs|4>                        # Number of epochs of training"
  echo "  --initial-learning-rate <initial-learning-rate|0.0002> # Learning rate at start of training"
  echo "  --final-learning-rate  <final-learning-rate|0.0004>   # Learning rate at end of training"
  echo "  --num-jobs-nnet <num-jobs|8>                     # Number of parallel jobs to use for main neural net"
  echo "                                                   # training (will affect results as well as speed; try 8, 16)"
  echo "                                                   # Note: if you increase this, you may want to also increase"
  echo "                                                   # the learning rate."
  echo "  --num-threads <num-threads|16>                   # Number of parallel threads per job (will affect results"
  echo "                                                   # as well as speed; may interact with batch size; if you increase"
  echo "                                                   # this, you may want to decrease the batch size."
  echo "                                                   # use multiple threads... note, you might have to reduce mem"
  echo "                                                   # versus your defaults, because it gets multiplied by the -pe smp argument."
  echo "  --mem <memory|1G>                                # memory requirement per thread, passed to queue.pl"
  echo "  --max-jobs-run <num-jobs|5>                      # Options given to e.g. queue.pl for jobs that do a lot of I/O."
  echo "  --samples-per-iter <#samples|200000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --stage <stage|-8>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  echo "  --criterion <criterion|smbr>                     # Training criterion: may be smbr, mmi or mpfe"
  echo "  --boost <boost|0.0>                              # Boosting factor for MMI (e.g., 0.1)"
  echo "  --modify-learning-rates <true,false|false>       # If true, modify learning rates to try to equalize relative"
  echo "                                                   # changes across layers."
  echo "  --uegs-dir <dir|"">                              # Directory for unsupervised discriminative examples, e.g. exp/foo/uegs"
  echo "  --degs-dir <dir|"">                              # Directory for supervised examples, e.g. exp/foo/egs"
  echo "  --online-ivector-dir <dir|"">                    # Directory for online-estimated iVectors, used in the"
  echo "  --src-model <model|"">                           # Use a difference model than \$srcdir/final.mdl"
  exit 1;
fi

data_sup=$1
data_unsup=$2
lang=$3
alidir=$4
denlatdir=$5
latdir=$6
srcdir=$7
dir=$8

[ -z "$src_model" ] && src_model=$srcdir/final.mdl
extra_files=
[ ! -z $online_ivector_dir_sup ] && \
 extra_files="$online_ivector_dir_sup/ivector_period $online_ivector_dir_sup/ivector_online.scp"
[ ! -z $online_ivector_dir_unsup ] && \
 extra_files="$online_ivector_dir_unsup/ivector_period $online_ivector_dir_unsup/ivector_online.scp"

# Check some files.
for f in $data_sup/feats.scp $data_unsup/feats.scp $lang/L.fst \
         $alidir/ali.1.gz $alidir/num_jobs \
         $denlatdir/lat.1.gz $latdir/num_jobs \
         $latdir/lat.1.gz $latdir/num_jobs $src_model \
         $srcdir/tree $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

nj_sup=$(cat $alidir/num_jobs) || exit 1; # caution: $nj_sup is the number of alignments

if ! [ $nj_sup == $(cat $denlatdir/num_jobs) ]; then
  echo "Number of jobs mismatch: $nj_sup versus $(cat $denlatdir/num_jobs)"
fi

nj_unsup=$(cat $latdir/num_jobs) || exit 1; # caution: $nj_unsup is the number of
                                      # splits of the lats, but
                                      # num_jobs_nnet is the number of nnet training
                                      # jobs we run in parallel.

mkdir -p $dir/log || exit 1;
[ -z "$uegs_dir" ] && mkdir -p $dir/uegs
[ -z "$degs_dir" ] && mkdir -p $dir/degs

sdata_sup=$data_sup/split$nj_sup
utils/split_data.sh $data_sup $nj_sup

sdata_unsup=$data_unsup/split$nj_unsup
utils/split_data.sh $data_unsup $nj_unsup

# function to remove egs that might be soft links.
remove () { for x in $*; do [ -L $x ] && rm $(readlink -f $x); rm $x; done }

splice_opts=`cat $alidir/splice_opts 2>/dev/null`
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
cp $alidir/splice_opts $dir 2>/dev/null
cp $alidir/cmvn_opts $dir 2>/dev/null
cp $alidir/tree $dir

const_dim_opt=
if [ ! -z "$online_ivector_dir_unsup" ]; then
  ivector_period=$(cat $online_ivector_dir_unsup/ivector_period)
  ivector_dim=$(feat-to-dim scp:$online_ivector_dir_unsup/ivector_online.scp -) || exit 1;
  # the 'const_dim_opt' allows it to write only one iVector per example,
  # rather than one per time-index... it has to average over
  const_dim_opt="--const-feat-dim=$ivector_dim"
fi

if [ ! -z "$transform_dir_sup" ] && [ ! -f $transform_dir_sup/trans.1 ]; then
  [ ! -f $transform_dir_sup/raw_trans.1 ] && echo "transform_dir_sup specified as $transform_dir_sup; but $transform_dir_sup/trans.1 or $transform_dir_sup/raw_trans.1 not found" && exit 1
fi

if [ ! -z "$transform_dir_unsup" ] && [ ! -f $transform_dir_unsup/trans.1 ]; then
  [ ! -f $transform_dir_unsup/raw_trans.1 ] && echo "transform_dir_unsup specified as $transform_dir_unsup; but $transform_dir_unsup/trans.1 or $transform_dir_unsup/raw_trans.1 not found" && exit 1
fi

if [ ! -z "$transform_dir_sup" ] && [ ! -f $transform_dir_sup/trans.1 ]; then
  [ ! -f $transform_dir_sup/raw_trans.1 ] && echo "transform_dir_sup specified as $transform_dir_sup; but $transform_dir_sup/trans.1 or $transform_dir_sup/raw_trans.1 not found" && exit 1
fi

if [ ! -z "$transform_dir_unsup" ] && [ ! -f $transform_dir_unsup/trans.1 ]; then
  [ ! -f $transform_dir_unsup/raw_trans.1 ] && echo "transform_dir_unsup specified as $transform_dir_unsup; but $transform_dir_unsup/trans.1 or $transform_dir_unsup/raw_trans.1 not found" && exit 1
fi

[ -z "$transform_dir_unsup" ] && echo "$0: --transform-dir-unsup was not specified. Trying $srcdir as transform_dir_unsup" && transform_dir_unsup=$srcdir
[ -z "$transform_dir_sup" ] && echo "$0 -- transform-dir-sup was not specified. Trying $alidir as transform_dir_sup" && transform_dir_sup=$alidir

## Set up features.
## Don't support deltas, only LDA or raw (mainly because deltas are less frequently used).
if [ -z "$feat_type" ]; then
  if [ -f $alidir/final.mat ] && [ ! -f $transform_dir_sup/raw_trans.1 ]; then feat_type=lda; else feat_type=raw; fi
fi
echo "$0: feature type is $feat_type"

# TODO: Make sure that the supervised data and the unsupervised data use the
# same kinds of transforms
case $feat_type in
  raw) feats_unsup="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata_unsup/JOB/utt2spk scp:$sdata_unsup/JOB/cmvn.scp scp:$sdata_unsup/JOB/feats.scp ark:- |"
    [ -f $alidir/final.mat ] && [ ! -f $transform_dir_unsup/raw_trans.1 ] && exit 1
    feats_sup="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata_sup/JOB/utt2spk scp:$sdata_sup/JOB/cmvn.scp scp:$sdata_sup/JOB/feats.scp ark:- |"
   ;;
  lda) 
    splice_opts=`cat $alidir/splice_opts 2>/dev/null`
    cp $alidir/final.mat $dir    
    feats_sup="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata_sup/JOB/utt2spk scp:$sdata_sup/JOB/cmvn.scp scp:$sdata_sup/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    feats_unsup="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata_unsup/JOB/utt2spk scp:$sdata_unsup/JOB/cmvn.scp scp:$sdata_unsup/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

if [ ! -z "$transform_dir_unsup" ]; then
  echo "$0: using transforms from $transform_dir_unsup"
  [ ! -s $transform_dir_unsup/num_jobs ] && \
    echo "$0: expected $transform_dir_unsup/num_jobs to contain the number of jobs." && exit 1;
  nj_orig=$(cat $transform_dir_unsup/num_jobs)
  if ! [ $nj_orig == $nj_unsup ]; then
    echo "Number of jobs mismatch: $nj_orig versus $nj_unsup"
    exit 1;
  fi

  if [ $feat_type == "raw" ]; then trans=raw_trans;
  else trans=trans; fi
  if [ $feat_type == "lda" ] && ! cmp `dirname $transform_dir_unsup`/final.mat $alidir/final.mat; then
    echo "$0: LDA transforms differ between $alidir and `dirname $transform_dir_unsup`"
    exit 1;
  fi

  feats_unsup="$feats_unsup transform-feats --utt2spk=ark:$sdata_unsup/JOB/utt2spk ark:$transform_dir_unsup/$trans.JOB ark:- ark:- |"
fi

if [ ! -z "$transform_dir_sup" ]; then
  echo "$0: using transforms from $transform_dir_sup"
  [ ! -s $transform_dir_sup/num_jobs ] && \
    echo "$0: expected $transform_dir_sup/num_jobs to contain the number of jobs." && exit 1;
  nj_orig=$(cat $transform_dir_sup/num_jobs)
  if ! [ $nj_orig == $nj_sup ]; then
    echo "Number of jobs mismatch: $nj_orig versus $nj_sup"
    exit 1;
  fi

  if [ $feat_type == "raw" ]; then trans=raw_trans;
  else trans=trans; fi
  if [ $feat_type == "lda" ] && ! cmp $transform_dir_sup/final.mat $alidir/final.mat; then
    echo "$0: LDA transforms differ between $alidir and $transform_dir_sup"
    exit 1;
  fi

  feats_sup="$feats_sup transform-feats --utt2spk=ark:$sdata_sup/JOB/utt2spk ark:$transform_dir_sup/$trans.JOB ark:- ark:- |"
fi

if [ ! -z $online_ivector_dir_sup ]; then
  # add iVectors to the features.
  feats_sup="$feats_sup paste-feats --length-tolerance=$ivector_period ark:-
  'ark,s,cs:utils/filter_scp.pl $sdata_sup/JOB/utt2spk $online_ivector_dir_sup/ivector_online.scp | subsample-feats --n=-$ivector_period scp:- ark:- |' ark:- |"
fi

if [ ! -z $online_ivector_dir_unsup ]; then
  feats_unsup="$feats_unsup paste-feats --length-tolerance=$ivector_period ark:-
  'ark,s,cs:utils/filter_scp.pl $sdata_unsup/JOB/utt2spk
  $online_ivector_dir_unsup/ivector_online.scp | subsample-feats --n=-$ivector_period scp:- ark:- |' ark:- |"
fi

num_frames_unsup=$(steps/nnet2/get_num_frames.sh $data_unsup)
num_frames_sup=$(steps/nnet2/get_num_frames.sh $data_sup)
num_jobs_nnet_sup=$[num_jobs_nnet/2]
num_jobs_nnet_unsup=$[num_jobs_nnet-num_jobs_nnet_sup]
iters_per_epoch=`perl -e "print int(2 * ($num_frames_unsup+$num_frames_unsup)/($samples_per_iter * $num_jobs_nnet) + 0.5);"` || exit 1;
[ $iters_per_epoch -eq 0 ] && iters_per_epoch=1
################################################################################
# Prepare Unsupervised examples
################################################################################

if [ -z "$uegs_dir" ]; then
  if [ $stage -le -9 ]; then
    echo "$0: working out number of frames of training data"
    echo $num_frames_unsup > $dir/num_frames_unsup
    # Working out number of iterations per epoch.
    echo $iters_per_epoch > $dir/uegs/iters_per_epoch  || exit 1;
  else
    num_frames_unsup=$(cat $dir/num_frames_unsup) || exit 1;
    iters_per_epoch=$(cat $dir/uegs/iters_per_epoch) || exit 1;
  fi

  samples_per_iter_real=`perl -e "print int($num_frames_unsup/($num_jobs_nnet_unsup * $iters_per_epoch))"`
  echo "$0: Every epoch, splitting the data up into $iters_per_epoch iterations,"
  echo "$0: giving samples-per-iteration of $samples_per_iter_real (you requested $[samples_per_iter * num_frames_unsup / (num_frames_sup+num_frames_unsup)])."
else
  iters_per_epoch=$(cat $uegs_dir/iters_per_epoch) || exit 1;
  [ -z "$iters_per_epoch" ] && exit 1;
  echo "$0: Every epoch, splitting the data up into $iters_per_epoch iterations"
fi

# we create these data links regardless of the stage, as there are situations where we
# would want to recreate a data link that had previously been deleted.
if [ -z "$uegs_dir" ] && [ -d $dir/uegs/storage ]; then
  echo "$0: creating data links for distributed storage of uegs"
    # See utils/create_split_dir.pl for how this 'storage' directory
    # is created.
  for x in $(seq $num_jobs_nnet_unsup); do
    for y in $(seq $nj_unsup); do
      utils/create_data_link.pl $dir/uegs/uegs_orig.$x.$y.ark || true
    done
    for z in $(seq 0 $[$iters_per_epoch-1]); do
      utils/create_data_link.pl $dir/uegs/uegs_tmp.$x.$z.ark || true
      utils/create_data_link.pl $dir/uegs/uegs.$x.$z.ark || true
    done
  done
fi

if [ $stage -le -8 ]; then
  # We want online preconditioning with a larger number of samples of history, since
  # in this setup the frames are only randomized at the segment level so they are highly
  # correlated.  It might make sense to tune this a little, later on, although I doubt
  # it matters once it's large enough.
  echo "$0: Copying initial model and removing any preconditioning"
  if $use_preconditioning; then
    $cmd $dir/log/convert.log \
      nnet-am-copy --learning-rate=$learning_rate "$src_model" - \| \
      nnet-am-switch-preconditioning  --num-samples-history=50000 - $dir/0.mdl || exit 1;
  else 
    $cmd $dir/log/convert.log \
      nnet-am-copy --learning-rate=$learning_rate "$src_model" $dir/0.mdl || exit 1;
  fi
fi


if [ $stage -le -7 ] && [ -z "$uegs_dir" ]; then
  echo "$0: getting initial training examples from lattices"
  
  egs_list=
  for n in `seq 1 $num_jobs_nnet_unsup`; do
    egs_list="$egs_list ark:$dir/uegs/uegs_orig.$n.JOB.ark"
  done

  $cmd --max-jobs-run $max_jobs_run JOB=1:$nj_unsup $dir/log/get_unsupervised_egs.JOB.log \
    nnet-get-egs-discriminative-unsupervised \
    $dir/0.mdl "$feats_unsup" \
    "ark,s,cs:gunzip -c $latdir/lat.JOB.gz|" ark:- \| \
    nnet-copy-egs-discriminative-unsupervised $const_dim_opt ark:- $egs_list || exit 1;
fi

if [ $stage -le -6 ] && [ -z "$uegs_dir" ]; then
  echo "$0: rearranging examples into parts for different parallel jobs"

  # combine all the "egs_orig.JOB.*.scp" (over the $nj_unsup splits of the data) and
  # then split into multiple parts egs.JOB.*.scp for different parts of the
  # data, 0 .. $iters_per_epoch_unsup-1.

  if [ $iters_per_epoch -eq 1 ]; then
    echo "Since iters-per-epoch == 1, just concatenating the data."
    for n in `seq 1 $num_jobs_nnet_unsup`; do
      cat $dir/uegs/uegs_orig.$n.*.ark > $dir/uegs/uegs_tmp.$n.0.ark || exit 1;
      remove $dir/uegs/uegs_orig.$n.*.ark  # don't "|| exit 1", due to NFS bugs...
    done
  else # We'll have to split it up using nnet-copy-egs.
    egs_list=
    for n in `seq 0 $[$iters_per_epoch-1]`; do
      egs_list="$egs_list ark:$dir/uegs/uegs_tmp.JOB.$n.ark"
    done
    # note, the "|| true" below is a workaround for NFS bugs
    # we encountered running this script with Debian-7, NFS-v4.
    $cmd --max-jobs-run $max_jobs_run JOB=1:$num_jobs_nnet_unsup $dir/log/split_egs.JOB.log \
      nnet-copy-egs-discriminative-unsupervised --srand=JOB \
        "ark:cat $dir/uegs/uegs_orig.JOB.*.ark|" $egs_list || exit 1
    remove $dir/uegs/uegs_orig.*.*.ark 
  fi
fi


if [ $stage -le -5 ] && [ -z "$uegs_dir" ]; then
  # Next, shuffle the order of the examples in each of those files.
  # Each one should not be too large, so we can do this in memory.
  # Then combine the examples together to form suitable-size minibatches
  # (for discriminative examples, it's one example per minibatch, so we
  # have to combine the lattices).
  echo "Shuffling the order of training examples"
  echo "(in order to avoid stressing the disk, these won't all run at once)."

  # note, the "|| true" below is a workaround for NFS bugs
  # we encountered running this script with Debian-7, NFS-v4.
  for n in `seq 0 $[$iters_per_epoch-1]`; do
    $cmd --max-jobs-run $max_jobs_run JOB=1:$num_jobs_nnet_unsup $dir/log/shuffle.$n.JOB.log \
      nnet-shuffle-egs-discriminative-unsupervised "--srand=\$[JOB+($num_jobs_nnet_unsup*$n)]" \
      ark:$dir/uegs/uegs_tmp.JOB.$n.ark ark:- \| \
      nnet-copy-egs-discriminative-unsupervised ark:- ark:$dir/uegs/uegs.JOB.$n.ark || exit 1
    remove $dir/uegs/uegs_tmp.*.$n.ark 
  done
fi

if [ -z "$uegs_dir" ]; then
  uegs_dir=$dir/uegs
fi

num_iters=$[$num_epochs * $iters_per_epoch];

if [ -z "$degs_dir" ]; then
  if [ $stage -le -4 ]; then
    echo "$0: working out number of frames of training data"
    echo $num_frames_sup > $dir/num_frames_sup
    # Working out number of iterations per epoch.
    echo $iters_per_epoch > $dir/degs/iters_per_epoch  || exit 1;
  else
    num_frames_sup=$(cat $dir/num_frames_sup) || exit 1;
    iters_per_epoch=$(cat $dir/degs/iters_per_epoch) || exit 1;
  fi

  samples_per_iter_real=`perl -e "print int($num_frames_sup/($num_jobs_nnet_sup * $iters_per_epoch))"`
  echo "$0: Every epoch, splitting the supervised data up into $iters_per_epoch iterations,"
  echo "$0: giving samples-per-iteration of $samples_per_iter_real (you requested $[samples_per_iter * num_frames_sup / (num_frames_sup+num_frames_unsup)])."
else
  iters_per_epoch=$(cat $degs_dir/iters_per_epoch) || exit 1;
  [ -z "$iters_per_epoch" ] && exit 1;
  echo "$0: Every epoch, splitting the data up into $iters_per_epoch iterations"
fi

if [ $stage -le -3 ] && [ -z "$degs_dir" ]; then
  echo "$0: getting initial training examples by splitting lattices"

  egs_list=
  for n in `seq 1 $num_jobs_nnet_sup`; do
    egs_list="$egs_list ark:$dir/degs/degs_orig.$n.JOB.ark"
  done


  $cmd --max-jobs-run $max_jobs_run JOB=1:$nj_sup $dir/log/get_egs.JOB.log \
    nnet-get-egs-discriminative --criterion=$criterion --drop-frames=$drop_frames \
     $dir/0.mdl "$feats_sup" \
    "ark,s,cs:gunzip -c $alidir/ali.JOB.gz |" \
    "ark,s,cs:gunzip -c $denlatdir/lat.JOB.gz|" ark:- \| \
    nnet-copy-egs-discriminative $const_dim_opt ark:- $egs_list || exit 1;
fi

if [ $stage -le -2 ] && [ -z "$degs_dir" ]; then
  echo "$0: rearranging examples into parts for different parallel jobs"

  # combine all the "egs_orig.JOB.*.scp" (over the $nj_sup splits of the data) and
  # then split into multiple parts egs.JOB.*.scp for different parts of the
  # data, 0 .. $iters_per_epoch-1.

  if [ $iters_per_epoch -eq 1 ]; then
    echo "Since iters-per-epoch == 1, just concatenating the data."
    for n in `seq 1 $num_jobs_nnet_sup`; do
      cat $dir/degs/degs_orig.$n.*.ark > $dir/degs/degs_tmp.$n.0.ark || exit 1;
      remove $dir/degs/degs_orig.$n.*.ark  # don't "|| exit 1", due to NFS bugs...
    done
  else # We'll have to split it up using nnet-copy-egs.
    egs_list=
    for n in `seq 0 $[$iters_per_epoch-1]`; do
      egs_list="$egs_list ark:$dir/degs/degs_tmp.JOB.$n.ark"
    done
    $cmd --max-jobs-run $max_jobs_run JOB=1:$num_jobs_nnet_sup $dir/log/split_egs.JOB.log \
      nnet-copy-egs-discriminative --srand=JOB \
        "ark:cat $dir/degs/degs_orig.JOB.*.ark|" $egs_list || exit 1;
    remove $dir/degs/degs_orig.*.*.ark
  fi
fi

if [ $stage -le -1 ] && [ -z "$degs_dir" ]; then
  # Next, shuffle the order of the examples in each of those files.
  # Each one should not be too large, so we can do this in memory.
  # Then combine the examples together to form suitable-size minibatches
  # (for discriminative examples, it's one example per minibatch, so we
  # have to combine the lattices).
  echo "Shuffling the order of training examples"
  echo "(in order to avoid stressing the disk, these won't all run at once)."

  # note, the "|| true" below is a workaround for NFS bugs
  # we encountered running this script with Debian-7, NFS-v4.
  # Also, we should note that we used to do nnet-combine-egs-discriminative
  # at this stage, but if iVectors are used this would expand the size of
  # the examples on disk (because they could no longer be stored in the spk_info
  # variable of the discrminative example, no longer being constant), so
  # now we do the nnet-combine-egs-discriminative operation on the fly during
  # training.
  for n in `seq 0 $[$iters_per_epoch-1]`; do
    $cmd --max-jobs-run $max_jobs_run JOB=1:$num_jobs_nnet_sup $dir/log/shuffle.$n.JOB.log \
      nnet-shuffle-egs-discriminative "--srand=\$[JOB+($num_jobs_nnet_sup*$n)]" \
      ark:$dir/degs/degs_tmp.JOB.$n.ark ark:$dir/degs/degs.JOB.$n.ark || exit 1;
    remove $dir/degs/degs_tmp.*.$n.ark
  done
fi

if [ -z "$degs_dir" ]; then
  degs_dir=$dir/degs
fi

echo "$0: Will train for $num_epochs epochs = $num_iters iterations"

if [ $num_threads -eq 1 ]; then
  train_suffix="-simple" # this enables us to use GPU code if
                         # we have just one thread.
  num_gpu=1
else
  train_suffix="-parallel --num-threads=$num_threads"
  num_gpu=0
fi

x=0   
while [ $x -lt $num_iters ]; do
  if [ $x -ge 0 ] && [ $stage -le $x ]; then
    
    echo "Training neural net (pass $x)"

    (
    $cmd --num-threads $num_threads --mem $mem --num-gpu $num_gpu JOB=1:$num_jobs_nnet_sup $dir/log/train_sup.$x.JOB.log \
      nnet-train-discriminative$train_suffix --silence-phones=$silphonelist \
      --criterion=$criterion --drop-frames=$drop_frames \
      --boost=$boost --acoustic-scale=$acoustic_scale \
      $dir/$x.mdl "ark:nnet-combine-egs-discriminative ark:$degs_dir/degs.JOB.$[$x%$iters_per_epoch].ark ark:- |" \
      $dir/$[$x+1].JOB.mdl &
    $cmd --num-threads $num_threads --mem $mem --num-gpu $num_gpu JOB=1:$num_jobs_nnet_unsup $dir/log/train_unsup.$x.JOB.log \
      nnet-train-discriminative-unsupervised$train_suffix \
      --acoustic-scale=$acoustic_scale --verbose=2 \
      "nnet-am-copy --learning-rate-factor=$unsupervised_scale $dir/$x.mdl - |" \
      ark:$uegs_dir/uegs.JOB.$[$x%$iters_per_epoch].ark $dir/$[$x+1].\$[JOB+$num_jobs_nnet_sup].mdl &
    wait || exit 1
    )  

    nnets_list=
    for n in `seq 1 $[num_jobs_nnet_sup+num_jobs_nnet_unsup]`; do
      [ -f $dir/$[$x+1].$n.mdl ] && nnets_list="$nnets_list $dir/$[$x+1].$n.mdl"
    done

    $cmd $dir/log/average.$x.log \
      nnet-am-average $nnets_list $dir/$[$x+1].mdl || exit 1;

    if $modify_learning_rates; then
      $cmd $dir/log/modify_learning_rates.$x.log \
        nnet-modify-learning-rates --retroactive=$retroactive \
        --last-layer-factor=$last_layer_factor \
        --first-layer-factor=$first_layer_factor \
        $dir/$x.mdl $dir/$[$x+1].mdl $dir/$[$x+1].mdl || exit 1;
    fi

    rm $nnets_list
  fi

  x=$[$x+1]
done

rm $dir/final.mdl 2>/dev/null || true
ln -s $x.mdl $dir/final.mdl
#if [ $stage -le $[$num_iters+1] ]; then
#  echo "Getting average posterior for purposes of adjusting the priors."
#  # Note: this just uses CPUs, using a smallish subset of data.
#  rm $dir/post.*.vec 2>/dev/null || true
#  $cmd JOB=1:$num_jobs_nnet $dir/log/get_post.JOB.log \
#    nnet-subset-egs --n=$prior_subset_size ark:$egs_dir/combine.egs ark:- \| \
#    nnet-compute-from-egs "nnet-to-raw-nnet $dir/$x.mdl -|" ark:- ark:- \| \
#    matrix-sum-rows ark:- ark:- \| vector-sum ark:- $dir/post.JOB.vec || exit 1;
#
#  sleep 3;  # make sure there is time for $dir/post.*.vec to appear.
#
#  $cmd $dir/log/vector_sum.log \
#   vector-sum $dir/post.*.vec $dir/post.vec || exit 1;
#
#  rm $dir/post.*.vec || true;
#
#  echo "Re-adjusting priors based on computed posteriors"
#  $cmd $dir/log/adjust_priors.log \
#    nnet-adjust-priors $dir/$x.mdl $dir/post.vec $dir/final.mdl || exit 1;
#fi

sleep 2

echo Done

if $cleanup; then
  echo Cleaning up data

  echo Removing training examples
  if [ -d $dir/uegs ] && [ ! -L $dir/uegs ]; then # only remove if directory is not a soft link.
    remove $dir/uegs/uegs.*
  fi

  echo Removing most of the models
  for x in `seq 0 $num_iters`; do
    if [ $[$x%$iters_per_epoch] -ne 0 ] || [ $[$x%$iters_per_epoch] -ne 0 ]; then
      # delete all but the epoch-final models.
      rm $dir/$x.mdl 2>/dev/null || true
    fi
  done
fi

for n in $(seq 0 $num_epochs); do
  x=$[$n*$iters_per_epoch]
  rm $dir/epoch$n.mdl 2>/dev/null || true
  ln -s $x.mdl $dir/epoch$n.mdl
done

