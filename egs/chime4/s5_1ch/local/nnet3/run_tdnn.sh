#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

# local/chime4_calc_wers.sh exp/nnet3_isolated_1ch_track/tdnn isolated_1ch_track exp/tri3b_tr05_multi_noisy/graph_tgpr_5k/
# compute dt05 WER for each location
#
# -------------------
# best overall dt05 WER 18.27% (language model weight = 10)
# -------------------
# dt05_simu WER: 17.98% (Average), 15.58% (BUS), 22.79% (CAFE), 14.03% (PEDESTRIAN), 19.54% (STREET)
# -------------------
# dt05_real WER: 18.56% (Average), 23.51% (BUS), 17.80% (CAFE), 12.75% (PEDESTRIAN), 20.19% (STREET)
# -------------------
#
stage=0
affix=
train_stage=-10
common_egs_dir=
reporting_email=
remove_egs=true

#chime4 specific options
train=noisy

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s <enhancement method>\n\n" `basename $0`
  echo "First argument specifies a unique name for different enhancement method"
  exit 1;
fi

# set enhanced data
enhan=$1

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# check whether run_init is executed
if [ ! -d data/lang ]; then
  echo "error, execute local/run_init.sh, first"
  exit 1;
fi

# check whether run_init is executed
if [ ! -d exp/tri3b_tr05_multi_${train} ]; then
  echo "error, execute local/run_init.sh, first"
  exit 1;
fi

local/nnet3/run_ivector_common.sh --stage $stage $enhan || exit 1;

# these values are set based on local/nnet3/run_ivector_common.sh
nnet3_dir=nnet3_$enhan
dir=exp/$nnet3_dir/tdnn
dir=$dir${affix:+_$affix}
train_set=tr05_multi_${train}_sp
ali_dir=exp/tri3b_${train_set}_ali

if [ $stage -le 8 ]; then
  echo "$0: creating neural net configs";

  # create the config files for nnet initialization
  steps/nnet3/tdnn/make_configs.py  \
    --feat-dir data/${train_set}_hires \
    --ivector-dir exp/$nnet3_dir/ivectors_${train_set} \
    --ali-dir $ali_dir \
    --relu-dim 256 \
    --subset-dim 128 \
    --splice-indexes "-1,0,1 -1,0,1 -3,0,3 -3,0,3 -3,0,3 -6,-3,0 0"  \
    --use-presoftmax-prior-scale true \
   $dir/configs || exit 1;
fi

if [ $stage -le 9 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/chime4-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir exp/$nnet3_dir/ivectors_${train_set} \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 6 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 4 \
    --trainer.optimization.initial-effective-lrate 0.0017 \
    --trainer.optimization.final-effective-lrate 0.00017 \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 10 \
    --feat-dir=data/${train_set}_hires \
    --ali-dir $ali_dir \
    --lang data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;

fi

graph_dir=exp/tri3b_tr05_multi_noisy/graph_tgpr_5k
if [ $stage -le 10 ]; then
  for decode_set in dt05_real_$enhan dt05_simu_$enhan; do
    (
    num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
    steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
        --online-ivector-dir exp/$nnet3_dir/ivectors_${decode_set} \
       $graph_dir data/${decode_set}_hires $dir/decode_tgpr_5k_${decode_set} || exit 1;
    ) &
  done
fi
wait;
exit 0;
