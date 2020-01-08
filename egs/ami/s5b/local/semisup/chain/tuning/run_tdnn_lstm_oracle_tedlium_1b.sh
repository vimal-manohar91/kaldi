#!/bin/bash

# This script is to demonstrate T-S learning using out-of-domain
# unsupervised data from Tedlium corpus to improve ASR on AMI-IHM.

set -e -o pipefail -u

# configs for 'chain'
stage=-10
train_stage=-10
get_egs_stage=-10
nj=80
max_jobs_run=10   # max number of parallel IO jobs

exp_root=exp/semisup_tedlium_ihm

# seed model params -- Used for decoding the clean data
src_dir=exp/ihm/chain_cleaned_rvb/tdnn_lstm1b6_sp_rvb_bi/
treedir=exp/ihm/chain_cleaned/tree_bi
ivector_extractor=exp/ihm/nnet3_cleaned_rvb/extractor  # used for decoding the unsupervised clean data
unsup_decode_opts="--extra-left-context 50 --frames-per-chunk 160"  # depends on src_dir

sup_data_dir=data/ihm/train_cleaned_sp_rvb_hires
sup_lat_dir=exp/ihm/chain_cleaned_rvb/tri3_cleaned_train_cleaned_sp_rvb_lats

sup_ivector_dir=exp/ihm/nnet3_cleaned_rvb/ivectors_train_cleaned_sp_rvb_hires    # If not supplied, i-vectors will be extracted using the ivector_extractor

# Unsupervised clean and (parallel) noisy data
unsup_data_dir=data/tedlium_train

# lang for decoding unsupervised data
unsup_graph_affix=_tedlium
unsup_lang=data/tedlium_lang_nosp

# Phone LM weights for den.fst: AMI (sup), Tedlium (unsup) weight
lm_weights=1,1

supervision_weights=1,1   # Supervision weights: AMI, Tedlium (headset), Tedlium (array)
num_copies=1,1    # Make copies of data: AMI, Tedlium (headset), Tedlium (array)

tdnn_affix=_1b_oracle
chain_affix=_semisup_ami_ihm_tedlium
nnet3_affix=_semisup_ami_ihm_tedlium

# neural network opts
hidden_dim=1024
cell_dim=1024
projection_dim=256

# training options
num_epochs=4
chunk_left_context=40
chunk_right_context=0
label_delay=5
dropout_schedule='0,0@0.20,0.3@0.50,0' # dropout schedule controls the dropout
                                       # proportion for each training iteration.
xent_regularize=0.025

# egs options to skip egs generation stages
remove_egs=false
sup_egs_dir=
unsup_egs_dir=
sup_frames_per_eg=160,140,110,80
unsup_frames_per_eg=150

lattice_lm_scale=0.5
unsup_egs_opts=""   # Extra opts for get_egs for unsupervised data
train_opts=   # Extra opts for train.py

# lang for decoding AMI test data
test_graph_affix=
test_lang=data/lang_ami_fsh.o3g.kn.pr1-7

# Decoding opts
extra_left_context=50
extra_right_context=0
frames_per_chunk_decoding=160

decode_iter=  # Iteration to decode with

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

unsup_dataset=$(basename $unsup_data_dir)

unsup_lat_dir=${src_dir}_lats_${unsup_dataset}_sp   # training lattices directory

dir=$exp_root/chain${chain_affix}/tdnn_lstm${tdnn_affix}_sp

lang=data/lang_chain

for f in $ivector_extractor/final.ie $treedir/final.mdl $src_dir/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
    exit 1
  fi
done

utils/lang/check_phones_compatible.sh $lang/phones.txt data/lang/phones.txt || exit 1
utils/lang/check_phones_compatible.sh $test_lang/phones.txt data/lang/phones.txt || exit 1
utils/lang/check_phones_compatible.sh $unsup_lang/phones.txt data/lang/phones.txt || exit 1
utils/lang/check_phones_compatible.sh $treedir/phones.txt data/lang/phones.txt || exit 1
utils/lang/check_phones_compatible.sh $src_dir/phones.txt data/lang/phones.txt || exit 1

steps/nnet2/check_ivectors_compatible.sh $ivector_extractor $src_dir || exit 1
diff $treedir/tree $src_dir/tree || exit 1

# ################################################################################
# # Prepare supervised "noisy" data directory and lattices
# ################################################################################
# 
# sup_tgt_data_dir=data/sdm1/train_cleaned_for_combine_sp_hires
# sup_tgt_lat_dir=exp/ihm/chain_cleaned/tri3_cleaned_train_cleaned_for_combine_sp_lats
# 
# if [ $stage -le -6 ]; then
#   local/semisup/copy_lat_dir_ihm.sh --nj $nj --cmd "$train_cmd" \
#     $sup_tgt_data_dir_ihmdata \
#     $sup_tgt_data_dir \
#     $sup_tgt_lat_dir_ihmdata \
#     $sup_tgt_lat_dir
# fi

################################################################################
# Extract features for the unsupervised source-domain (clean) data
################################################################################

unsup_src_ivector_dir=$(dirname $ivector_extractor)/ivectors_${unsup_dataset}_sp

if [ $stage -le -5 ]; then
  utils/data/perturb_data_dir_speed_3way.sh $unsup_data_dir \
    ${unsup_data_dir}_sp_hires
  utils/data/perturb_data_dir_volume.sh ${unsup_data_dir}_sp_hires

  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd" --nj $nj \
    ${unsup_data_dir}_sp_hires
  steps/compute_cmvn_stats.sh ${unsup_data_dir}_sp_hires
  utils/fix_data_dir.sh ${unsup_data_dir}_sp_hires

  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    ${unsup_data_dir}_sp_hires ${unsup_data_dir}_sp_hires_max2
fi 

if [ $stage -le -4 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${unsup_data_dir}_sp_hires_max2 $ivector_extractor \
    $unsup_src_ivector_dir
fi

unsup_data_dir_sp=${unsup_data_dir}_sp_hires

################################################################################
# Decode unsupervised source-domain (clean) data using
# unsupervised data's LM and source-domain acoustic model
################################################################################

if [ $stage -le -2 ]; then
  steps/nnet3/align_lats.sh --nj $nj --cmd "$decode_cmd" \
    --scale-opts "--transition-scale=1.0 --self-loop-scale=1.0" \
    --acoustic-scale 1.0 --generate-ali-from-lats true \
    $unsup_decode_opts \
    --online-ivector-dir $unsup_src_ivector_dir \
    ${unsup_data_dir_sp} $unsup_lang $src_dir $unsup_lat_dir || exit 1
fi

################################################################################
# Extract i-vectors for target-domain
################################################################################

unsup_ivector_dir=$(dirname $ivector_extractor)/ivectors_${unsup_dataset}_rvb

if [ $stage -le 1 ]; then
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    ${unsup_data_dir_sp} \
    ${unsup_data_dir_sp}_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${unsup_data_dir_sp}_max2 \
    $ivector_extractor \
    $unsup_ivector_dir
fi

mkdir -p $dir

if [ $stage -le 7 ]; then
  steps/nnet3/chain/make_weighted_den_fst.sh --num-repeats $lm_weights \
    --cmd "$train_cmd" \
    $treedir $unsup_lat_dir \
    $dir
fi

if [ $stage -le 9 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  tdnn_opts="l2-regularize=0.0015"
  lstm_opts="l2-regularize=0.0006 decay-time=20 dropout-proportion=0.0"
  output_opts="l2-regularize=0.00025"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=$hidden_dim $tdnn_opts
  relu-renorm-layer name=tdnn2 input=Append(-1,0,1) dim=$hidden_dim $tdnn_opts
  relu-renorm-layer name=tdnn3 input=Append(-1,0,1) dim=$hidden_dim $tdnn_opts

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmp-layer name=lstm1 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 $lstm_opts
  relu-renorm-layer name=tdnn4 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  relu-renorm-layer name=tdnn5 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  relu-renorm-layer name=tdnn6 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  fast-lstmp-layer name=lstm2 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 $lstm_opts
  relu-renorm-layer name=tdnn7 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  relu-renorm-layer name=tdnn8 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  relu-renorm-layer name=tdnn9 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  fast-lstmp-layer name=lstm3 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 $lstm_opts

  ## adding the layers for chain branch
  output-layer name=output input=lstm3 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5 $output_opts

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=lstm3 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5 $output_opts
  output name=output-0 input=output.affine@$label_delay 
  output name=output-1 input=output.affine@$label_delay 

  output name=output-0-xent input=output-xent.log-softmax@$label_delay 
  output name=output-1-xent input=output-xent.log-softmax@$label_delay 
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

frame_subsampling_factor=1
if [ -f $src_dir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $src_dir/frame_subsampling_factor) || exit 1
fi

. $dir/configs/vars

left_context=$[model_left_context + chunk_left_context]
right_context=$[model_right_context + chunk_right_context]
left_context_initial=$model_left_context
right_context_final=$model_right_context

egs_left_context=`perl -e "print int($left_context + $frame_subsampling_factor / 2)"`
egs_right_context=`perl -e "print int($right_context + $frame_subsampling_factor / 2)"`
egs_left_context_initial=`perl -e "print int($left_context_initial + $frame_subsampling_factor / 2)"`
egs_right_context_final=`perl -e "print int($right_context_final + $frame_subsampling_factor / 2)"`

cmvn_opts=`cat $src_dir/cmvn_opts` || exit 1

sup_dataset=$(basename $sup_data_dir)

if [ -z "$sup_egs_dir" ]; then
  sup_egs_dir=$dir/egs_${sup_dataset}

  if [ $stage -le 10 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5c/$sup_egs_dir/storage $sup_egs_dir/storage
    fi

    steps/nnet3/chain/get_egs.sh --cmd "$decode_cmd" \
      --left-context $egs_left_context --right-context $egs_right_context \
      --left-context-initial $egs_left_context_initial --right-context-final $egs_right_context_final \
      --frame-subsampling-factor $frame_subsampling_factor \
      --alignment-subsampling-factor $frame_subsampling_factor \
      --frames-per-eg $sup_frames_per_eg \
      --frames-per-iter 1500000 \
      --cmvn-opts "$cmvn_opts" \
      --online-ivector-dir $sup_ivector_dir \
      --generate-egs-scp true \
      $sup_data_dir $dir $sup_lat_dir $sup_egs_dir
    touch $sup_egs_dir/.nodelete # keep egs around when that run dies.
  fi
else
  sup_frames_per_eg=$(cat $sup_egs_dir/info/frames_per_eg)
fi

if [ -z "$unsup_egs_dir" ]; then
  unsup_egs_dir=$dir/egs_${unsup_dataset}_sp
  if [ $stage -le 11 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $unsup_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$unsup_egs_dir/storage $unsup_egs_dir/storage
    fi

    steps/nnet3/chain/get_egs.sh --cmd "$decode_cmd" --alignment-subsampling-factor 1 \
      --left-tolerance 1 --right-tolerance 1 \
      --left-context $egs_left_context --right-context $egs_right_context \
      --left-context-initial $egs_left_context_initial --right-context-final $egs_right_context_final \
      --frames-per-eg $unsup_frames_per_eg --frames-per-iter 1500000 \
      --frame-subsampling-factor $frame_subsampling_factor \
      --cmvn-opts "$cmvn_opts" \
      --online-ivector-dir $unsup_ivector_dir \
      --generate-egs-scp true $unsup_egs_opts \
      ${unsup_data_dir_sp} $dir \
      $unsup_lat_dir $unsup_egs_dir

    touch $unsup_egs_dir/.nodelete
  fi
fi

if [ $stage -le 13 ]; then
  steps/nnet3/chain/multilingual/combine_egs.sh --cmd "$train_cmd" \
    --block-size 128 \
    --lang2weight $supervision_weights --lang2num-copies "$num_copies" \
    2 $sup_egs_dir $unsup_egs_dir \
    $dir/egs_comb
fi

if [ $train_stage -le -4 ]; then
  train_stage=-4
fi

if [ $stage -le 14 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$train_cmd --mem 4G" --train-queue-opt "--h-rt 00:20:00" --combine-queue-opt "--h-rt 00:59:00" \
    --feat.online-ivector-dir $sup_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights true \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$dir/egs_comb" \
    --egs.opts "--frames-overlap-per-eg 0 --generate-egs-scp true" \
    --chain.right-tolerance 1 --chain.left-tolerance 1 \
    --chain.alignment-subsampling-factor 1 \
    --egs.chunk-width $sup_frames_per_eg \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.num-chunk-per-minibatch 64,32 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
    --trainer.deriv-truncate-margin 8 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $sup_data_dir \
    --tree-dir $treedir \
    --lat-dir $sup_lat_dir \
    --dir $dir $train_opts || exit 1;
fi

graph_dir=$dir/graph${test_graph_affix}
cp $sup_egs_dir/tree $dir

if [ $stage -le 15 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 ${test_lang} $dir $graph_dir
fi

if [ $stage -le 17 ]; then
  rm -f $dir/.error
  for dset in dev eval; do
    (
      decode_dir=$dir/decode${test_graph_affix}_${dset}

      steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --extra-left-context $extra_left_context \
        --extra-right-context $extra_right_context \
        --extra-left-context-initial 0 --extra-right-context-final 0 \
        --frames-per-chunk $frames_per_chunk_decoding \
        --online-ivector-dir $(dirname $ivector_extractor)/ivectors_${dset}_hires \
        $graph_dir data/$mic/${dset}_hires $decode_dir || { echo "Failed decoding in $decode_dir"; touch $dir/.error; }
    ) &
  done
  wait

  if [ -f $dir/.error ]; then
    echo "Failed decoding."
    exit 1
  fi
fi

decode_tedlium=true
if $decode_tedlium; then
  if [ $stage -le 18 ]; then
    for data in tedlium_dev tedlium_test; do
      utils/copy_data_dir.sh data/${data}{,_hires}
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 30 --mfcc-config conf/mfcc_hires.conf data/${data}_hires
      steps/compute_cmvn_stats.sh data/${data}_hires
      steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
        data/${data}_hires $ivector_extractor \
        $(dirname $ivector_extractor)/ivectors_${data}_hires
    done
  fi

  test_lang=data/tedlium_lang_nosp
  test_graph_affix=_tedlium
  graph_dir=$dir/graph${test_graph_affix}
  if [ $stage -le 19 ]; then
    # Note: it might appear that this $lang directory is mismatched, and it is as
    # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
    # the lang directory.
    utils/mkgraph.sh --self-loop-scale 1.0 ${test_lang} $dir $graph_dir
  fi

  nnet3-am-copy --edits="remove-output-nodes name=output;rename-node old-name=output-1 new-name=output" $dir/final.mdl $dir/final_output_1.mdl || exit 1
  if [ $stage -le 20 ]; then
    rm -f $dir/.error
    for dset in tedlium_dev tedlium_test; do
      (
      decode_dir=$dir/decode${test_graph_affix}_${dset}_iterfinal_output_1

      steps/nnet3/decode.sh --nj 30 --cmd "$decode_cmd" \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --extra-left-context $extra_left_context \
        --extra-right-context $extra_right_context \
        --extra-left-context-initial 0 --extra-right-context-final 0 \
        --frames-per-chunk $frames_per_chunk_decoding --skip-scoring true --iter final_output_1 \
        --online-ivector-dir $(dirname $ivector_extractor)/ivectors_${dset}_hires \
        $graph_dir data/${dset}_hires $decode_dir || { echo "Failed decoding in $decode_dir"; touch $dir/.error; }
      ) &
    done
    wait

    if [ -f $dir/.error ]; then
      echo "Failed decoding."
      exit 1
    fi
  fi

  if [ $stage -le 21 ]; then
    for dset in tedlium_dev tedlium_test; do
      decode_dir=$dir/decode${test_graph_affix}_${dset}_iterfinal_output_1
      steps/scoring/score_kaldi_wer.sh --min-lmwt 8 --max-lmwt 12 \
        --cmd "$decode_cmd" data/${dset}_hires \
        $graph_dir $decode_dir
    done
  fi
fi
