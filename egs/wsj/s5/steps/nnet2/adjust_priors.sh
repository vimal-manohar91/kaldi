#!/bin/bash

# Copyright 2014  Vimal Manohar
# Apache 2.0.


# This screen updates the priors of a neural-network model.

# Begin configuration section.
cmd=run.pl
prior_subset_size=10000 # 10k samples per job, for computing priors.  Should be
                        # more than enough.
iter=final
out_model=final

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <egs-dir> <exp-dir>"
  echo " e.g.: $0 exp/tri4_nnet/egs exp/tri4_nnet"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --prior-subset-size <int|10000>                  # number of egs per job for prior adjustment"
  echo "  --iter <input-model-iteration|final>             # uses \$iter.mdl as the input model"
  echo "  --out-model <output-model|final>                 # write to \$out_model.mdl"
  exit 1
fi

egs_dir=$1
dir=$2

rm $dir/post.*.vec 2>/dev/null

num_archives=`cat $egs_dir/info/num_archives` || exit 1

nj=`perl -e "print $num_archives >= 24 ? 24 : $num_archives"`

$cmd JOB=1:$nj $dir/log/get_post.JOB.log \
  nnet-subset-egs --n=$prior_subset_size ark:$egs_dir/egs.JOB.ark ark:- \| \
  nnet-compute-from-egs "nnet-to-raw-nnet $dir/$iter.mdl -|" ark:- ark:- \| \
  matrix-sum-rows ark:- ark:- \| vector-sum ark:- $dir/post.JOB.vec || exit 1

sleep 3;
      
run.pl $dir/log/vector_sum.log \
  vector-sum $dir/post.*.vec $dir/post.vec || exit 1

rm $dir/post.*.vec
      
echo "Re-adjusting priors based on computed posteriors"
run.pl $dir/log/adjust_priors.$iter.log \
  nnet-adjust-priors $dir/$iter.mdl $dir/post.vec $dir/$out_model.mdl || exit 1
