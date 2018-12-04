#!/bin/bash

# This script does MMI + KL training.
# This script is similar to _d, but combines the supevised data, both
# IHM-reverberated and SDM to a single output to match the baseline.
# This script supports generates numerator posteriors
# after splitting egs.
# This script does not use dropout for comparison 
# with the baseline that also does not use dropout.
set -e -o pipefail -u

# configs for 'chain'
stage=-10
train_stage=-10
get_egs_stage=-10
nj=80
max_jobs_run=30

exp_root=exp/semisup_icsi

# seed model params -- Used for decoding the clean data
src_dir=exp/ihm/chain_cleaned_rvb/tdnn_lstm1i_sp_rvb_bi/
treedir=exp/ihm/chain_cleaned/tree_bi
src_ivector_extractor=exp/ihm/nnet3_cleaned_rvb/extractor  # used for decoding the unsup source-domain data

# These are good, but the utt-ids for SDM need to be fixed
sup_data_dir=data/sdm1/train_cleaned_sp_rvb_hires
sup_lat_dir=exp/sdm1/chain_cleaned_rvb/tri3_cleaned_train_cleaned_sp_rvb_lats_ihmdata

# Use the same i-vector extractor trained on AMI
tgt_ivector_extractor=exp/sdm1/nnet3_cleaned_rvb/extractor
sup_ivector_dir=exp/sdm1/nnet3_cleaned_rvb/ivectors_train_cleaned_sp_rvb_hires

unsup_decode_opts="--extra-left-context 50 --frames-per-chunk 160"  # depends on src_dir

# Unsupervised clean and noisy data
unsup_src_data_dir=data/train_icsi_ihm
unsup_tgt_data_dir=data/train_icsi_sdm_all

tgt_mfcc_config=conf/mfcc_hires.conf

unsup_graph_affix=_fixed
unsup_lang=data/lang_ami_fsh.o3g.kn.pr1-7

supervision_weights=1,1,1
num_copies=4,3,1  # There is 4x ICSI SDM data
                  # We reduce a little of ICSI IHM data to let SDM be prominent

test_graph_affix=
test_lang=data/lang_ami_fsh.o3g.kn.pr1-7

lm_weights=2,1   # AMI (sup), ICSI (unsup) weight

tdnn_affix=_1e
chain_affix=_semisup_ami_icsi

kl_factor_schedule="output-0=0,0 output-1=0,0 output-2=0,0"
mmi_factor_schedule="output-0=1,1 output-1=1,1 output-2=1,1"

hidden_dim=1024
cell_dim=1024
projection_dim=256

# training options
num_epochs=0.18
minibatch_size=64,32
chunk_left_context=40
chunk_right_context=0
label_delay=5

remove_egs=false
sup_egs_dir=
unsup_src_egs_dir=
unsup_tgt_egs_dir=
unsup_frames_per_eg=150

lattice_lm_scale=0.5
kl_fst_scale=0.5
unsup_egs_opts=""
train_opts=

scoring_script=local/score.sh

extra_left_context=50
extra_right_context=0

# End configuration section.
echo "$0 $@"  # Print the command line for logging

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

unsup_src_dataset=$(basename $unsup_src_data_dir)
unsup_tgt_dataset=$(basename $unsup_tgt_data_dir)

unsup_src_lat_dir=$src_dir/decode${unsup_graph_affix}_${unsup_src_dataset}_sp   # training lattices directory
unsup_tgt_lat_dir=$src_dir/decode${unsup_graph_affix}_${unsup_tgt_dataset}_sp   # training lattices directory

dir=$exp_root/chain${chain_affix}/tdnn_lstm${tdnn_affix}

lang=data/lang_chain

for f in $src_ivector_extractor/final.ie $treedir/final.mdl $src_dir/final.mdl \
  $tgt_mfcc_config; do
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

steps/nnet2/check_ivectors_compatible.sh $src_ivector_extractor $src_dir || exit 1
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

unsup_src_ivector_dir=$(dirname $src_ivector_extractor)/ivectors_${unsup_src_dataset}_sp

if [ $stage -le -4 ]; then
  utils/data/perturb_data_dir_speed_3way.sh $unsup_src_data_dir \
    ${unsup_src_data_dir}_sp_hires
  utils/data/perturb_data_dir_volume.sh ${unsup_src_data_dir}_sp_hires

  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd" --nj $nj \
    ${unsup_src_data_dir}_sp_hires
  steps/compute_cmvn_stats.sh ${unsup_src_data_dir}_sp_hires
  utils/fix_data_dir.sh ${unsup_src_data_dir}_sp_hires

  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    ${unsup_src_data_dir}_sp_hires ${unsup_src_data_dir}_sp_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${unsup_src_data_dir}_sp_hires_max2 $src_ivector_extractor \
    $unsup_src_ivector_dir
fi

unsup_src_data_dir_sp=${unsup_src_data_dir}_sp_hires

################################################################################
# Decode unsupervised source-domain (clean) data using
# unsupervised data's LM and source-domain acoustic model
################################################################################

unsup_graph_dir=$src_dir/graph${unsup_graph_affix}

if [ $stage -le -3 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 ${unsup_lang} $src_dir $unsup_graph_dir
fi

if [ $stage -le -2 ]; then
  steps/nnet3/decode_semisup.sh --nj $nj --cmd "$decode_cmd" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --write-compact false --word-determinize false \
    $unsup_decode_opts \
    --frames-per-chunk 160 \
    --online-ivector-dir $unsup_src_ivector_dir \
    --skip-scoring true \
    $unsup_graph_dir ${unsup_src_data_dir_sp} $unsup_src_lat_dir || exit 1

  ln -sf ../final.mdl ${unsup_src_lat_dir}
fi

################################################################################
# Augment unsupervised source-domain (clean) data with RIRs and noises
################################################################################
unsup_src_data_dir_perturbed=${unsup_src_data_dir}_sp_rvb1_hires

if [ $stage -le -1 ]; then
  utils/copy_data_dir.sh ${unsup_src_data_dir_sp} ${unsup_src_data_dir}_sp

  rm -r $unsup_src_data_dir_perturbed 2>/dev/null || true

  local/nnet3/multi_condition/run_reverb_datadir.sh \
    --num-data-reps 1 \
    --norvb-data-dir ${unsup_src_data_dir}_sp

  rm -r ${unsup_src_data_dir}_sp
fi

################################################################################
# Get lattices for parallel target-domain data
################################################################################

unsup_src_lat_dir_rvb=${unsup_src_lat_dir}_rvb

if [ $stage -le 1 ]; then
  local/semisup/copy_lat_dir.sh --write-compact false \
    --cmd "$decode_cmd" --nj $nj --num-data-reps 1 \
    ${unsup_src_data_dir_perturbed} \
    $unsup_src_lat_dir $unsup_src_lat_dir_rvb || exit 1
  ln -sf ../final.mdl ${unsup_src_lat_dir_rvb}
fi

if [ $stage -le 2 ]; then
  utils/data/perturb_data_dir_speed_3way.sh $unsup_tgt_data_dir \
    ${unsup_tgt_data_dir}_sp_hires
  utils/data/perturb_data_dir_volume.sh ${unsup_tgt_data_dir}_sp_hires

  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd --max-jobs-run 30" --nj $nj \
    ${unsup_tgt_data_dir}_sp_hires
  steps/compute_cmvn_stats.sh ${unsup_tgt_data_dir}_sp_hires
  utils/fix_data_dir.sh ${unsup_tgt_data_dir}_sp_hires
fi

unsup_tgt_data_dir_perturbed=${unsup_tgt_data_dir}_sp_hires

if [ $stage -le 3 ]; then
  local/semisup/copy_lat_dir_icsi_parallel.sh --write-compact false \
    --cmd "$decode_cmd" --nj $nj \
    ${unsup_src_data_dir_sp} ${unsup_tgt_data_dir_perturbed} \
    $unsup_src_lat_dir $unsup_tgt_lat_dir || exit 1
  ln -sf ../final.mdl ${unsup_tgt_lat_dir}
fi

################################################################################
# Train i-vector extractor for target-domain
################################################################################

unsup_tgt_ivector_dir=$(dirname $tgt_ivector_extractor)/ivectors_${unsup_tgt_dataset}_rvb

if [ $stage -le 4 ]; then
  utils/combine_data.sh \
    ${unsup_tgt_data_dir}_rvb_hires \
    $unsup_src_data_dir_perturbed $unsup_tgt_data_dir_perturbed

  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    ${unsup_tgt_data_dir}_rvb_hires \
    ${unsup_tgt_data_dir}_rvb_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${unsup_tgt_data_dir}_rvb_hires_max2 \
    $tgt_ivector_extractor \
    $unsup_tgt_ivector_dir

  exit 3
fi

unsup_best_path_dir=$src_dir/best_path${unsup_graph_affix}_${unsup_src_dataset}_sp

if [ $stage -le 5 ]; then
  steps/best_path_weights.sh --cmd "$decode_cmd" \
    ${unsup_src_data_dir} ${unsup_src_lat_dir} \
    $src_dir/best_path${unsup_graph_affix}_${unsup_src_dataset}_sp
fi

mkdir -p $dir
deriv_weights_scp=$dir/unsup_deriv_weights.scp

if [ $stage -le 6 ]; then
  for n in `seq 3`; do
    cat $unsup_best_path_dir/weights.scp | awk -v n=$n '{print "rev"n"_"$0}'
  done | sort -k1,1 > $unsup_src_lat_dir_rvb/weights.scp

  for mic in sdm1 sdm2 sdm3 sdm4; do
    cat $unsup_best_path_dir/weights.scp | \
      utils/apply_map.pl -f 1 ${unsup_tgt_lat_dir}/ihmutt2utt.$mic
  done | sort -k1,1 > $unsup_tgt_lat_dir/weights.scp
fi

cat $unsup_src_lat_dir_rvb/weights.scp $unsup_tgt_lat_dir/weights.scp | \
  sort -k1,1 > $deriv_weights_scp

if [ $stage -le 7 ]; then
  steps/nnet3/chain/make_weighted_den_fst.sh --num-repeats $lm_weights \
    --cmd "$train_cmd" \
    $treedir $src_dir/best_path${unsup_graph_affix}_${unsup_src_dataset}_sp \
    $dir
fi

xent_regularize=0.1

if [ $stage -le 8 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  lstm_opts="decay-time=40"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=$hidden_dim
  relu-renorm-layer name=tdnn2 input=Append(-1,0,1) dim=$hidden_dim
  relu-renorm-layer name=tdnn3 input=Append(-1,0,1) dim=$hidden_dim

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  lstmp-layer name=lstm1 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3
  relu-renorm-layer name=tdnn4 input=Append(-3,0,3) dim=$hidden_dim
  relu-renorm-layer name=tdnn5 input=Append(-3,0,3) dim=$hidden_dim
  relu-renorm-layer name=tdnn6 input=Append(-3,0,3) dim=$hidden_dim
  lstmp-layer name=lstm2 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3
  relu-renorm-layer name=tdnn7 input=Append(-3,0,3) dim=$hidden_dim
  relu-renorm-layer name=tdnn8 input=Append(-3,0,3) dim=$hidden_dim
  relu-renorm-layer name=tdnn9 input=Append(-3,0,3) dim=$hidden_dim
  lstmp-layer name=lstm3 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3

  ## adding the layers for chain branch
  output-layer name=output input=lstm3 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=lstm3 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
  output name=output-0 input=output.affine@$label_delay 
  output name=output-1 input=output.affine@$label_delay 
  output name=output-2 input=output.affine@$label_delay 

  output name=output-0-xent input=output-xent.log-softmax@$label_delay 
  output name=output-1-xent input=output-xent.log-softmax@$label_delay 
  output name=output-2-xent input=output-xent.log-softmax@$label_delay 
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
  frames_per_eg=$(cat $src_dir/egs/info/frames_per_eg)

  if [ $stage -le 9 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5c/$sup_egs_dir/storage $sup_egs_dir/storage
    fi

    steps/nnet3/chain/get_egs.sh --cmd "$decode_cmd" \
      --left-context $egs_left_context --right-context $egs_right_context \
      --left-context-initial $egs_left_context_initial --right-context-final $egs_right_context_final \
      --frame-subsampling-factor $frame_subsampling_factor \
      --alignment-subsampling-factor $frame_subsampling_factor \
      --frames-per-eg $frames_per_eg \
      --frames-per-iter 1500000 \
      --cmvn-opts "$cmvn_opts" \
      --online-ivector-dir $sup_ivector_dir \
      --generate-egs-scp true \
      $sup_data_dir $dir $sup_lat_dir $sup_egs_dir
    touch $sup_egs_dir/.nodelete # keep egs around when that run dies.
  fi
else
  frames_per_eg=$(cat $sup_egs_dir/info/frames_per_eg)
fi

unsup_egs_opts="$unsup_egs_opts --deriv-weights-scp $deriv_weights_scp"

if [ -z "$unsup_src_egs_dir" ]; then
  unsup_src_egs_dir=$dir/egs_${unsup_src_dataset}_sp_rvb1
  [ -z $unsup_frames_per_eg ] && unsup_frames_per_eg=$frames_per_eg
  if [ $stage -le 11 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $unsup_src_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$unsup_src_egs_dir/storage $unsup_src_egs_dir/storage
    fi

    steps/nnet3/chain/get_egs_split.sh --cmd "$decode_cmd" --alignment-subsampling-factor 1 \
      --left-tolerance 1 --right-tolerance 1 \
      --left-context $egs_left_context --right-context $egs_right_context \
      --left-context-initial $egs_left_context_initial --right-context-final $egs_right_context_final \
      --frames-per-eg $unsup_frames_per_eg --frames-per-iter 1500000 \
      --frame-subsampling-factor $frame_subsampling_factor \
      --cmvn-opts "$cmvn_opts" --lattice-lm-scale $lattice_lm_scale \
      --lattice-prune-beam 4.0 --kl-latdir $unsup_src_lat_dir_rvb --kl-fst-scale $kl_fst_scale \
      --deriv-weights-scp $deriv_weights_scp \
      --online-ivector-dir $unsup_tgt_ivector_dir \
      --generate-egs-scp true $unsup_egs_opts \
      $unsup_src_data_dir_perturbed $dir \
      $unsup_src_lat_dir_rvb $unsup_src_egs_dir

    touch $unsup_src_egs_dir/.nodelete
  fi
fi

if [ -z "$unsup_tgt_egs_dir" ]; then
  unsup_tgt_egs_dir=$dir/egs_${unsup_tgt_dataset}_sp
  [ -z $unsup_frames_per_eg ] && unsup_frames_per_eg=$frames_per_eg
  if [ $stage -le 12 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $unsup_tgt_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$unsup_tgt_egs_dir/storage $unsup_tgt_egs_dir/storage
    fi

    steps/nnet3/chain/get_egs_split.sh --cmd "$decode_cmd" --alignment-subsampling-factor 1 \
      --left-tolerance 1 --right-tolerance 1 \
      --left-context $egs_left_context --right-context $egs_right_context \
      --left-context-initial $egs_left_context_initial --right-context-final $egs_right_context_final \
      --frames-per-eg $unsup_frames_per_eg --frames-per-iter 1500000 \
      --frame-subsampling-factor $frame_subsampling_factor \
      --cmvn-opts "$cmvn_opts" --lattice-lm-scale $lattice_lm_scale \
      --lattice-prune-beam 4.0 --kl-latdir $unsup_tgt_lat_dir --kl-fst-scale $kl_fst_scale \
      --deriv-weights-scp $deriv_weights_scp \
      --online-ivector-dir $unsup_tgt_ivector_dir \
      --generate-egs-scp true $unsup_egs_opts \
      $unsup_tgt_data_dir_perturbed $dir \
      $unsup_tgt_lat_dir $unsup_tgt_egs_dir

    touch $unsup_tgt_egs_dir/.nodelete
  fi
fi

if [ $stage -le 13 ]; then
  steps/nnet3/chain/multilingual/combine_egs.sh --cmd "$train_cmd" \
    --block-size 128 \
    --lang2weight $supervision_weights --lang2num-copies "$num_copies" \
    3 $sup_egs_dir $unsup_src_egs_dir $unsup_tgt_egs_dir \
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
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights true \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1500000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.deriv-truncate-margin 8 \
    --chain.kl-factor-schedule "$kl_factor_schedule" \
    --chain.mmi-factor-schedule "$mmi_factor_schedule" \
    --egs.stage $get_egs_stage --egs.get-egs-script "steps/nnet3/chain/get_egs_split.sh" \
    --egs.opts "--frames-overlap-per-eg 0 --generate-egs-scp true $unsup_egs_opts --max-jobs-run $max_jobs_run" \
    --chain.right-tolerance 1 --chain.left-tolerance 1 \
    --chain.alignment-subsampling-factor 1 \
    --egs.chunk-width 160,140,110,80 \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --egs.dir "$dir/egs_comb" \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $sup_data_dir \
    --tree-dir $treedir \
    --lat-dir $sup_lat_dir \
    --dir $dir $train_opts || exit 1;
fi

graph_dir=$dir/graph${test_graph_affix}
if [ $stage -le 15 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 ${test_lang} $dir $graph_dir
fi

if [ $stage -le 17 ]; then
  for dset in dev eval; do
    (
      decode_dir=$dir/decode${test_graph_affix}_${dset}

      steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --extra-left-context $extra_left_context \
        --extra-right-context $extra_right_context \
        --extra-left-context-initial 0 --extra-right-context-final 0 \
        --frames-per-chunk 160 \
        --online-ivector-dir $(dirname $tgt_ivector_extractor)/ivectors_${dset}_hires \
        $graph_dir data/sdm1/${dset}_hires $decode_dir || { echo "Failed decoding in $decode_dir"; touch $dir/.error; }
    ) &
  done
  wait

  if [ -f $dir/.error ]; then
    echo "Failed decoding."
    exit 1
  fi
fi
