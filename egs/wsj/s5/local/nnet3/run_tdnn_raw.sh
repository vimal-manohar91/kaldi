#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=10
train_stage=-10
cleanup=false
skip_lda=true
dir=exp/nnet3/nnet_tdnn_a
init_lr=0.001
final_lr=0.0001
egs_dir=
momentum=0.0
max_change=1.0
ivector_dir=
add_layers_period=5
init_job=2
final_job=14
relu_dim=750
num_relu=
low_rms=0.2
high_rms=0.2
target_rms=0.2


b_dim=
b_layer=1
splice_dim="-4,-3,-2,-1,0,1,2,3,4  0  -2,2  0  -4,4 0"
#splice_dim="-2,-1,0,1,2 -2,-1,0,1,2 -2,2 0 -4,4 0"
num_epochs=12
pnorm_input_dim=2000
pnorm_output_dim=250
shift_input=false
stretch_time=
add_log_sum=false
mb=512
# decode_opts
stop_rand=false # If true, it stops randomization component to randomize inputs.
iter=final

#ivector_dir=exp/nnet3/ivectors_train_si284
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

local/nnet3/run_ivector_common.sh --stage $stage || exit 1;


if [ ! -z "$relu_dim" ];then 
  opts="--relu-dim $relu_dim"
fi
if [ ! -z "$num_relu" ];then
  opts="$opts --num-relu $num_relu"
fi
if [ ! -z "$egs_dir" ];then
  opts="$opts --egs-dir $egs_dir"
fi

if [ ! -z "$b_dim" ];then 
  opts="$opts --bottleneck-dim $b_dim --bottleneck-layer $b_layer"
fi
if $shift_input; then
  if $num_epochs -le 20;then
    num_epochs=20
  fi
  opts="$opts --shift-input $shift_input"
fi
use_ivector=false
if [ "$ivector_dir" != "" ];then 
  use_ivector=true
  opts="$opts --online-ivector-dir $ivector_dir"
fi
if $stretch_time; then
  if [ $stage -le 8 ]; then
    #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
    # _sp stands for speed-perturbed

    for datadir in train_si284; do
      utils/perturb_data_dir_speed.sh 0.9 data/${datadir} data/temp1
      utils/perturb_data_dir_speed.sh 1.1 data/${datadir} data/temp2
      utils/combine_data.sh --extra-files utt2uniq data/${datadir}_tmp data/temp1 data/temp2
      rm -r data/temp1 data/temp2

      mfccdir=mfcc_perturbed
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
        data/${datadir}_tmp exp/make_mfcc/${datadir}_tmp $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh data/${datadir}_tmp exp/make_mfcc/${datadir}_tmp $mfccdir || exit 1;
      utils/fix_data_dir.sh data/${datadir}_tmp


      utils/copy_data_dir.sh --spk-prefix sp1.0- --utt-prefix sp1.0- data/${datadir} data/temp0
      utils/combine_data.sh data/${datadir}_sp data/${datadir}_tmp data/temp0
      rm -r data/temp0 data/${datadir}_tmp
    done
  fi

  if [ $stage -le 9 ]; then
    #obtain the alignment of the perturbed data
    steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
      data/train_si284_sp data/lang exp/tri4b_ali_si284 exp/tri4b_ali_si284_sp || exit 1
  fi
fi

if [ $stage -le 10 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,5,6}/$USER/kaldi-data/egs/wsj-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_tdnn${skip_lda:+"_raw"}.sh --stage $train_stage \
    --num-epochs $num_epochs  --num-jobs-initial 2 --num-jobs-final 14 \
    --splice-indexes "$splice_dim" \
    --feat-type raw \
    --shift-input $shift_input \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --io-opts "-tc 12" \
    --initial-effective-lrate $init_lr --final-effective-lrate $final_lr \
    --cmd "$decode_cmd" \
    --pnorm-input-dim $pnorm_input_dim \
    --pnorm-output-dim $pnorm_output_dim \
    --skip-lda $skip_lda \
    --momentum $momentum \
    --max-param-change $max_change \
    --add-layers-period $add_layers_period \
    --low-rms $low_rms --high-rms $high_rms $opts \
    --remove-egs false \
    --add-log-sum $add_log_sum \
    --minibatch-size $mb \
    --cleanup $cleanup \
    data/train_si284${stretch_time:+"_sp"} data/lang exp/tri4b_ali_si284${stretch_time:+"_sp"} $dir  || exit 1;
fi

echo "use_ivector = $use_ivector"

if [ $stage -le 11 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
  for lm_suffix in tgpr bd_tgpr; do
    graph_dir=exp/tri4b/graph_${lm_suffix}
    # use already-built graphs.
    for year in eval92 dev93; do
      if $use_ivector; then
        decode_opts="--online-ivector-dir exp/nnet3/ivectors_test_$year"
      fi
      steps/nnet3/decode_raw.sh --nj 8 --cmd "$decode_cmd" \
      $decode_opts --iter $iter \
      --low-rms $target_rms --high-rms $target_rms --stop-rand $stop_rand \
         $graph_dir data/test_${year}_hires $dir/decode_${lm_suffix}_${year} || exit 1;
    done
  done
fi


exit 0;

# results:
grep WER exp/nnet3/nnet_tdnn_a/decode_{tgpr,bd_tgpr}_{eval92,dev93}/scoring_kaldi/best_wer
exp/nnet3/nnet_tdnn_a/decode_tgpr_eval92/scoring_kaldi/best_wer:%WER 6.03 [ 340 / 5643, 74 ins, 20 del, 246 sub ] exp/nnet3/nnet_tdnn_a/decode_tgpr_eval92/wer_13_1.0
exp/nnet3/nnet_tdnn_a/decode_tgpr_dev93/scoring_kaldi/best_wer:%WER 9.35 [ 770 / 8234, 162 ins, 84 del, 524 sub ] exp/nnet3/nnet_tdnn_a/decode_tgpr_dev93/wer_11_0.5
exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_eval92/scoring_kaldi/best_wer:%WER 3.81 [ 215 / 5643, 30 ins, 18 del, 167 sub ] exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_eval92/wer_10_1.0
exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_dev93/scoring_kaldi/best_wer:%WER 6.74 [ 555 / 8234, 69 ins, 72 del, 414 sub ] exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_dev93/wer_11_0.0
b03:s5:
