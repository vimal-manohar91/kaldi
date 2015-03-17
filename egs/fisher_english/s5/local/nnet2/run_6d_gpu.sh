#!/bin/bash


# This script demonstrates discriminative training of neural nets.  It's on top
# of run_5c_gpu.sh, which uses adapted 40-dimensional features.  This version of
# the script uses GPUs.  We distinguish it by putting "_gpu" at the end of the
# directory name.


gpu_opts="--gpu 1"                   # This is suitable for the CLSP network,
                                      # you'll likely have to change it.  we'll
                                      # use it later on, in the training (it's
                                      # not used in denlat creation)
stage=0
train_stage=-100
srcdir=exp/nnet5c_gpu_i3000_o300_n4
num_threads=1
nj=32
criterion=smbr
boost=0.1
drop_frames=true
degs_dir=
learning_rate=9e-4
train_suffix=_100k
dir=
one_silence_class=true

set -e # exit on error.

. ./cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
. utils/parse_options.sh

if [ -z "$dir" ]; then
  dir=$(echo ${srcdir}_${train_suffix} | sed "s:5c_gpu:6c_gpu:")
fi
dir=${dir}_${criterion}_lr${learning_rate}

if [ "$criterion" == "mmi" ]; then
  dir=${dir}_b${boost}
  if ! $drop_frames; then
    dir=${dir}_nodrop
  fi
fi

if ! $one_silence_class; then
  dir=${dir}_no_onesilence
fi

# The denominator lattice creation currently doesn't use GPUs.

# Note: we specify 1G each for the mem_free and ram_free which, is per
# thread... it will likely be less than the default.  Increase the beam relative
# to the defaults; this is just for this RM setup, where the default beams will
# likely generate very thin lattices.  Note: the transform-dir is important to
# specify, since this system is on top of fMLLR features.

if [ $stage -le 0 ]; then
  steps/nnet2/align.sh  --cmd "$decode_cmd $gpu_opts" \
    --use-gpu yes \
    --transform-dir exp/tri4a_ali${train_suffix} \
    --nj $nj data/train${train_suffix} data/lang ${srcdir} ${srcdir}_ali${train_suffix}
fi

if [ $stage -le 1 ]; then
  steps/nnet2/make_denlats.sh --cmd "$decode_cmd --mem 1G" \
    --nj $nj --sub-split 20 --num-threads 6 --parallel-opts "--num-threads 6" \
    --transform-dir exp/tri4a_ali${train_suffix} --text data/train_100k/text \
    data/train${train_suffix} data/lang $srcdir ${srcdir}_denlats${train_suffix}
fi

if [ -z "$degs_dir" ]; then
  if [ $stage -le 2 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $degs_dir/storage ]; then
      utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/fisher_english-$(date +'%d_%m_%H_%M')/$dir/degs $dir/degs/storage 
    fi

    steps/nnet2/get_egs_discriminative2.sh --cmd "$decode_cmd --max-jobs-run 5" \
      --criterion smbr --drop-frames false \
      --transform-dir exp/tri4a_ali${train_suffix} \
      data/train${train_suffix} data/lang_100k_test \
      ${srcdir}_ali${train_suffix} ${srcdir}_denlats${train_suffix} $srcdir/final.mdl $dir/degs
  fi
fi

[ -z "$degs_dir" ] && degs_dir=$dir/degs

if [ $stage -le 3 ]; then
  steps/nnet2/train_discriminative2.sh \
    --cmd "$decode_cmd $gpu_opts"  --learning-rate $learning_rate \
    --modify-learning-rates true --last-layer-factor 0.1 \
    --num-epochs 4 --cleanup false \
    --num-jobs-nnet 4 --stage $train_stage \
    --num-threads $num_threads \
    --criterion $criterion --boost $boost --drop-frames $drop_frames \
    --src-model "$srcdir/final.mdl" --one-silence-class $one_silence_class \
    $degs_dir $dir
fi

if [ $stage -le 4 ]; then
  for epoch in 1 2 3 4; do 
    steps/nnet2/decode.sh --cmd "$decode_cmd --num-threads 6" \
      --nj 30 --iter epoch$epoch \
      --num-threads 6 \
      --config conf/decode.config --transform-dir exp/tri4a/decode_100k_dev \
      exp/tri4a/graph_100k data/dev $dir/decode_100k_dev_epoch$epoch &
  done
fi

exit 0;
