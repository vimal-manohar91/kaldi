#!/bin/bash

set -e 
set -o pipefail

# this is run_discriminative.sh

# This script does discriminative training on top of nnet3 system.
# note: this relies on having a cluster that has plenty of CPUs as well as GPUs,
# since the lattice generation runs in about real-time, so takes of the order of
# 1000 hours of CPU time.
# 
# Note: rather than using any features we have dumped on disk, this script
# regenerates them from the wav data three times-- when we do lattice
# generation, numerator alignment and discriminative training.  This made the
# script easier to write and more generic, because we don't have to know where
# the features and the iVectors are, but of course it's a little inefficient.
# The time taken is dominated by the lattice generation anyway, so this isn't
# a huge deal.

. cmd.sh


stage=0
train_stage=-10
get_egs_stage=-10
use_gpu=true
srcdir=exp/nnet2_online/nnet_ms_a
criterion=smbr
drop_frames=false  # only matters for MMI.
frames_per_eg=150
frame_overlap_per_eg=30
effective_learning_rate=0.0000125
num_jobs_nnet=4
train_stage=-10 # can be used to start training in the middle.
decode_start_epoch=0 # can be used to avoid decoding all epochs, e.g. if we decided to run more.
num_epochs=4
degs_dir=
cleanup=false  # run with --cleanup true --stage 6 to clean up (remove large things like denlats,
               # alignments and degs).
train_data_dir=data/train_nodup_sp_hires
online_ivector_dir=exp/nnet3/ivectors_train_nodup_sp
one_silence_class=false

set -e
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  num_threads=1
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
fi

if [ ! -f ${srcdir}/final.mdl ]; then
  echo "$0: expected ${srcdir}/final.mdl to exist; first run run_tdnn.sh or run_lstm.sh"
  exit 1;
fi


if [ $stage -le 1 ]; then
  nj=50  # this doesn't really affect anything strongly, except the num-jobs for one of
         # the phases of get_egs_discriminative2.sh below.
  num_threads_denlats=6
  subsplit=40 # number of jobs that run per job (but 2 run at a time, so total jobs is 80, giving
              # total slots = 80 * 6 = 480.
  steps/nnet3/make_denlats.sh --cmd "$decode_cmd --mem 1G --num-threads $num_threads_denlats" \
      --online-ivector-dir $online_ivector_dir \
      --nj $nj --sub-split $subsplit --num-threads "$num_threads_denlats" --config conf/decode.config \
     $train_data_dir data/lang $srcdir ${srcdir}_denlats || exit 1;

  # the command below is a more generic, but slower, way to do it.
  #steps/online/nnet2/make_denlats.sh --cmd "$decode_cmd -l mem_free=1G,ram_free=1G -pe smp $num_threads_denlats" \
  #    --nj $nj --sub-split $subsplit --num-threads "$num_threads_denlats" --config conf/decode.config \
  #   data/train_960 data/lang ${srcdir}_online ${srcdir}_denlats || exit 1;

fi

if [ $stage -le 2 ]; then
  # hardcode no-GPU for alignment, although you could use GPU [you wouldn't
  # get excellent GPU utilization though.]
  nj=350 # have a high number of jobs because this could take a while, and we might
         # have some stragglers.
  use_gpu=no
  gpu_opts=

  steps/nnet3/align.sh  --cmd "$decode_cmd $gpu_opts" --use-gpu "$use_gpu" \
     --online-ivector-dir $online_ivector_dir \
     --nj $nj $train_data_dir data/lang $srcdir ${srcdir}_ali || exit 1;

  # the command below is a more generic, but slower, way to do it.
  # steps/online/nnet2/align.sh --cmd "$decode_cmd $gpu_opts" --use-gpu "$use_gpu" \
  #    --nj $nj data/train_960 data/lang ${srcdir}_online ${srcdir}_ali || exit 1;
fi

left_context=14
right_context=10

if [ -z "$degs_dir" ]; then
  degs_dir=${srcdir}_degs
  if [ $stage -le 3 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${srcdir}_degs/storage ]; then
      utils/create_split_dir.pl \
        /export/b0{1,2,5,6}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5/${srcdir}_degs/storage ${srcdir}_degs/storage
    fi
    # have a higher maximum num-jobs if
    if [ -d ${srcdir}_degs/storage ]; then max_jobs=10; else max_jobs=5; fi

    steps/nnet3/get_egs_discriminative.sh \
      --cmd "$decode_cmd --max-jobs-run $max_jobs --mem 20G" --stage $get_egs_stage \
      --online-ivector-dir $online_ivector_dir --left-context $left_context --right-context $right_context \
      --criterion $criterion --frames-per-eg $frames_per_eg --frames-overlap-per-eg $frames_overlap_per_eg \
      $train_data_dir data/lang ${srcdir}{_ali,_denlats,/final.mdl,_degs} || exit 1;

    # the command below is a more generic, but slower, way to do it.
    #steps/online/nnet2/get_egs_discriminative2.sh \
      #  --cmd "$decode_cmd --max-jobs-run $max_jobs" \
      #  --criterion $criterion --drop-frames $drop_frames \
      #   data/train_960 data/lang ${srcdir}{_ali,_denlats,_online,_degs} || exit 1;
  fi
fi
  
dir=${srcdir}_${criterion}_${effective_learning_rate}

if ! $one_silence_class; then
  dir=${dir}_noonesil
fi

if [ $stage -le 4 ]; then
  steps/nnet3/train_discriminative.sh --cmd "$decode_cmd" \
    --stage $train_stage \
    --effective-lrate $effective_learning_rate \
    --criterion $criterion --drop-frames $drop_frames \
    --num-epochs $num_epochs --one-silence-class $one_silence_class \
    --num-jobs-nnet $num_jobs_nnet --num-threads $num_threads \
      ${degs_dir} $dir || exit 1
fi

graph_dir=exp/tri4/graph_sw1_tg

if [ $stage -le 5 ]; then
  for decode_set in train_dev eval2000; do
    (
    num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
    steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
        --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
       $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_hires_sw1_tg || exit 1;
    if $has_fisher; then
	steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
          data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
	  $dir/decode_${decode_set}_hires_sw1_{tg,fsh_fg} || exit 1;
    fi
    ) &
  done
fi
wait;

if [ $stage -le 6 ] && $cleanup; then
  # if you run with "--cleanup true --stage 6" you can clean up.
  rm ${srcdir}_denlats/lat.*.gz || true
  rm ${srcdir}_ali/ali.*.gz || true
  steps/nnet2/remove_egs.sh ${srcdir}_degs || true
fi


exit 0;

