#!/bin/bash


# This script demonstrates discriminative training of neural nets.  It's on top
# of run_5c_gpu.sh, which uses adapted 40-dimensional features.  This version of
# the script uses GPUs.  We distinguish it by putting "_gpu" at the end of the
# directory name.


gpu_opts="-l gpu=1"                   # This is suitable for the CLSP network,
                                      # you'll likely have to change it.  we'll
                                      # use it later on, in the training (it's
                                      # not used in denlat creation)
stage=0
train_stage=-100
srcdir=exp/nnet5c_gpu_i3000_o300_n4
nj=30

set -e # exit on error.

. ./cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
. utils/parse_options.sh


# The denominator lattice creation currently doesn't use GPUs.

# Note: we specify 1G each for the mem_free and ram_free which, is per
# thread... it will likely be less than the default.  Increase the beam relative
# to the defaults; this is just for this RM setup, where the default beams will
# likely generate very thin lattices.  Note: the transform-dir is important to
# specify, since this system is on top of fMLLR features.

if [ $stage -le 0 ]; then
  steps/nnet2/align.sh  --cmd "$decode_cmd $gpu_opts" \
    --use-gpu yes \
    --transform-dir exp/tri4a \
    --nj $nj data/train_100k data/lang ${srcdir} ${srcdir}_ali_100k
fi

if [ $stage -le 1 ]; then
  steps/nnet2/make_denlats.sh --cmd "$decode_cmd -l mem_free=1G,ram_free=1G" \
    --nj $nj --sub-split 20 --num-threads 6 --parallel-opts "-pe smp 6" \
    --transform-dir exp/tri4a \
    data/train_100k data/lang $srcdir ${srcdir}_denlats
fi

dir=$(echo $srcdir | sed 's:5c_gpu:6c_mpe_gpu:')
if [ $stage -le 2 ]; then
  steps/nnet2/train_discriminative.sh \
    --cmd "$decode_cmd"  --learning-rate 0.000002 \
    --modify-learning-rates true --last-layer-factor 0.1 \
    --num-epochs 4 --cleanup false \
    --num-jobs-nnet 4 --stage $train_stage \
    --transform-dir exp/tri4a \
    --num-threads 1 --parallel-opts "$gpu_opts" \
    data/train_100k data/lang \
    ${srcdir}_ali_100k ${srcdir}_denlats $srcdir/final.mdl $dir 
fi

if [ $stage -le 3 ]; then
  for epoch in 1 2 3 4; do 
    steps/nnet2/decode.sh --cmd "$decode_cmd" \
      --nj 30 --iter epoch$epoch \
      --parallel-opts "-pe smp 6" --num-threads 6 \
      --config conf/decode.config --transform-dir exp/tri4a/decode_100k_dev \
      exp/tri4a/graph_100k data/dev $dir/decode_100k_dev_epoch$epoch &
  done
fi

exit 0;
