#!/bin/bash

# Copyright   2013  Daniel Povey
#             2015  Vimal Manohar
# Apache 2.0.

# This script was modified from steps/online/nnet2/dump_nnet_activations.sh.
# Unlike that script, this script works on non-online nnet2 networks.  It is to
# be used when retraining the top layer of a system that was trained on another,
# out-of-domain dataset, on some in-domain dataset.  It processes the features
# of the out-of-domain data then puts it through all but the last layer of the
# neural net in that directory, and dumps those final activations in a feats.scp
# file in the output directory.  These files might be quite large.  A typical
# feature-dimension is 300; it's the p-norm output dim.  We compress these files
# (note: the compression is lossy).


# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
ivector_scale=1.0
iter=final
transform_dir=
feat_type=

online_ivector_dir=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <data> <srcdir> <output-dir>"
  echo " e.g.: $0 data/train exp/nnet2_online/nnet_a_online exp/nnet2_online/activations_train"
  echo "Output is in <output-dir>/feats.scp"
  echo "main options (for others, see top of script file)"
  echo "  --transform-dir <decoding-dir>           # directory of previous decoding"
  echo "                                           # where we can find transforms for SAT systems."
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue-opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "                                        # Set to 1 if compatibility with utterance-by-utterance"
  echo "                                        # decoding is the only factor, and to larger if you care "
  echo "                                        # also about adaptation over several utterances."
  echo "  --iter <iter>                            # Iteration of model to
  compute feats."
  exit 1;
fi

data=$1
srcdir=$2
dir=$3

[ ! -z "$online_ivector_dir" ] && \
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"

for f in $data/feats.scp $srcdir/$iter.mdl $extra_files; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

sdata=$data/split$nj;
cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1;

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs


## Set up features.
if [ -z "$feat_type" ]; then
  if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=raw; fi
  echo "$0: feature type is $feat_type"
fi

splice_opts=`cat $srcdir/splice_opts 2>/dev/null`

case $feat_type in
  raw) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -s $transform_dir/num_jobs ] && \
    echo "$0: expected $transform_dir/num_jobs to contain the number of jobs." && exit 1;
  nj_orig=$(cat $transform_dir/num_jobs)
  
  if [ $feat_type == "raw" ]; then trans=raw_trans;
  else trans=trans; fi
  if [ $feat_type == "lda" ] && \
    ! cmp $transform_dir/../final.mat $srcdir/final.mat && \
    ! cmp $transform_dir/final.mat $srcdir/final.mat; then
    echo "$0: LDA transforms differ between $srcdir and $transform_dir"
    exit 1;
  fi
  if [ ! -f $transform_dir/$trans.1 ]; then
    echo "$0: expected $transform_dir/$trans.1 to exist (--transform-dir option)"
    exit 1;
  fi
  if [ $nj -ne $nj_orig ]; then
    # Copy the transforms into an archive with an index.
    for n in $(seq $nj_orig); do cat $transform_dir/$trans.$n; done | \
       copy-feats ark:- ark,scp:$dir/$trans.ark,$dir/$trans.scp || exit 1;
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk scp:$dir/$trans.scp ark:- ark:- |"
  else
    # number of jobs matches with alignment dir.
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/$trans.JOB ark:- ark:- |"
  fi
elif grep 'transform-feats --utt2spk' $srcdir/log/train.1.log >&/dev/null; then
  echo "$0: **WARNING**: you seem to be using a neural net system trained with transforms,"
  echo "  but you are not providing the --transform-dir option in test time."
fi
##

if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  # note: subsample-feats, with negative n, will repeat each feature -n times.
  feats="$feats paste-feats --length-tolerance=$ivector_period ark:- 'ark,s,cs:utils/filter_scp.pl $sdata/JOB/utt2spk $online_ivector_dir/ivector_online.scp | subsample-feats --n=-$ivector_period scp:- ark:- | copy-matrix --scale=$ivector_scale ark:- ark:-|' ark:- |"
fi

mkdir -p $dir/feats $dir/data

if [ $stage -le 1 ]; then
  info=$dir/nnet_info
  nnet-am-info $srcdir/$iter.mdl >$info
  nc=$(grep num-components $info | awk '{print $2}');
  if grep SumGroupComponent $info >/dev/null; then 
    nc_truncate=$[$nc-3]  # we did mix-up: remove AffineComponent,
                          # SumGroupComponent, SoftmaxComponent
  else
    nc_truncate=$[$nc-2]  # remove AffineComponent, SoftmaxComponent
  fi
  nnet-to-raw-nnet --truncate=$nc_truncate $srcdir/$iter.mdl $dir/nnet.raw
fi

if [ $stage -le 2 ]; then
  echo "$0: dumping neural net activations"

  # The next line is a no-op unless $dir/feats/storage/ exists; see utils/create_split_dir.pl.
  for j in $(seq $nj); do  utils/create_data_link.pl $dir/feats/feats.$j.ark; done

  $cmd JOB=1:$nj $dir/log/dump_activations.JOB.log \
    nnet-compute $dir/nnet.raw "$feats" ark:- \| \
    copy-feats --compress=true ark:- \
    ark,scp:$dir/feats/feats.JOB.ark,$dir/feats/feats.JOB.scp || exit 1;
fi

if [ $stage -le 3 ]; then
  echo "$0: combining activations across jobs"
  mkdir -p $dir/data
  cp -r $data/* $dir/data
  for j in $(seq $nj); do cat $dir/feats/feats.$j.scp; done >$dir/data/feats.scp || exit 1;
fi

if [ $stage -le 4 ]; then
  echo "$0: computing [fake] CMVN stats."
  # We shouldn't actually be doing CMVN, but the get_egs.sh script expects it,
  # so create fake CMVN stats.
  steps/compute_cmvn_stats.sh --fake $dir/data $dir/log $dir/feats || exit 1
fi

echo "$0: done.  Output is in $dir/data/feats.scp"

