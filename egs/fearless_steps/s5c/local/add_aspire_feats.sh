#!/bin/bash

# Begin configuration section.
stage=1
nj=4 # number of decoding jobs.
cmd=run.pl
frames_per_chunk=50
iter=final
use_gpu=false # If true, will use a GPU
normalize_mean=true
normalize_variance=true
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
online_ivector_dir=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. utils/parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
  echo "e.g.:   steps/nnet3/decode.sh --nj 8 \\"
  echo "--online-ivector-dir exp/nnet2_online/ivectors_test_eval92 \\"
  echo "    exp/tri4b/graph_bg data/test_eval92_hires $dir/decode_bg_eval92"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                   # config containing options"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --iter <iter>                            # Iteration of model to decode; default is final."
  echo "  --scoring-opts <string>                  # options to local/score.sh"
  echo "  --num-threads <n>                        # number of threads to use, default 1."
  echo "  --use-gpu <true|false>                   # default: false.  If true, we recommend"
  echo "                                           # to use large --num-threads as the graph"
  echo "                                           # search becomes the limiting factor."
  exit 1;
fi

data=$1
dir=$2
aspire_feats=$3
dataout=$4
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.
model=$srcdir/final.mdl


extra_files=
if [ ! -z "$online_ivector_dir" ]; then
  steps/nnet2/check_ivectors_compatible.sh $srcdir $online_ivector_dir || exit 1
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"
fi

for f in $data/feats.scp $model $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

sdata=$data/split$nj;
cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1;
thread_string=
if $use_gpu; then
  thread_string=" --use-gpu=yes"
  queue_opt="--gpu 1"
else
  thread_string=" --use-gpu=no"
  queue_opt=""
fi

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs


## Set up features.
echo "$0: feature type is raw"


feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
fi


modeldim=$(nnet3-am-info "$model" | grep num-pdfs | sed 's/.*: *//g')
if [ $stage -le 1 ]; then
  utils/copy_data_dir.sh $data $aspire_feats/
fi

if [ $stage -le 2 ]; then
  #the dct in the following command is just to speed up the PCA computation
	echo "Applying the pca matrix"
  $cmd $queue_opt JOB=1:$nj $dir/log/apply_pca.JOB.log \
    nnet3-compute $thread_string $ivector_opts \
     --frames-per-chunk=$frames_per_chunk \
     --extra-left-context=$extra_left_context \
     --extra-right-context=$extra_right_context \
     --extra-left-context-initial=$extra_left_context_initial \
     --extra-right-context-final=$extra_right_context_final \
      "$model" "$feats" ark:- \| \
		transform-feats $dir/final.mat ark:- ark,scp:$aspire_feats/feats.JOB.ark,$aspire_feats/feats.JOB.scp
fi

if [ $stage -le 3 ]; then
  utils/copy_data_dir.sh $data $aspire_feats
  cat $aspire_feats/feats.*.scp | sort > $aspire_feats/feats.scp
  steps/paste_feats.sh $data $aspire_feats $dataout $dataout/log $dataout/feats
  steps/compute_cmvn_stats.sh $dataout
  utils/fix_data_dir.sh $dataout
fi

exit 0;
