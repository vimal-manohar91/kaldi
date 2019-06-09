#!/bin/bash

set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
train_data_dir=data/apollo11_whole_legal_1a_seg
src_dir=exp/chain/tdnn_lstm_1a
extractor=exp/nnet3/extractor
treedir=exp/chain/tree_bi_a
src_lang=data/lang
lang_test=data/lang_test

chunk_left_context=40
chunk_right_context=0
dropout_schedule='0,0@0.20,0.3@0.50,0'
xent_regularize=0.025
label_delay=5

nj=100

nnet3_affix=
train_stage=-10
affix=_1b
get_egs_stage=-10

# Unsupervised opts
frames_per_eg=150  # Using a frames-per-eg of 150 for unsupervised data
                         # was found to be better than allowing smaller chunks
                         # (160,140,110,80) like for supervised system
lattice_lm_scale=0.5  # lm-scale for using the weights from unsupervised lattices when
                      # creating numerator supervision
lattice_prune_beam=4.0  # beam for pruning the lattices prior to getting egs
                        # for unsupervised data
tolerance=1   # frame-tolerance for chain training

kl_fst_scale=0.0

mmi_factor_schedule="output=1,1"
kl_factor_schedule="output=0,0"

# Neural network opts
hidden_dim=1024
cell_dim=1024
projection_dim=256

common_egs_dir=
xent_regularize=0.025
primary_lr_factor=1.0
phone_lm_scales=0,1
num_epochs=2

# training options
srand=0
remove_egs=false
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

utils/lang/check_phones_compatible.sh $src_lang/phones.txt $src_dir/phones.txt || exit 1
utils/lang/check_phones_compatible.sh $src_lang/phones.txt $treedir/phones.txt || exit 1
utils/lang/check_phones_compatible.sh $src_lang/phones.txt $lang_test/phones.txt || exit 1

diff $treedir/tree $src_dir/tree || \
  { echo "$treedir/tree and $src_dir/tree are different!" && exit 1; }

if [ $stage -le 0 ]; then
  if [ -f ${train_data_dir}_hires/feats.scp ]; then
    echo "$0: ${train_data_dir}_hires/feats.scp exits. Remove it or skip this stage."
    exit 1
  fi

  utils/copy_data_dir.sh ${train_data_dir} ${train_data_dir}_hires
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf \
    --cmd "$train_cmd" --nj $nj \
    ${train_data_dir}_hires
  steps/compute_cmvn_stats.sh ${train_data_dir}_hires
  utils/fix_data_dir.sh ${train_data_dir}_hires
fi

dir=${src_dir}_adapt_all${affix}

exit 0

train_id=$(basename $train_data_dir)
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_id}

if [ $stage -le 1 ]; then
  steps/online/nnet2/extract_ivectors_online.sh \
    --cmd "$train_cmd" --nj $nj \
    ${train_data_dir}_hires $extractor \
    exp/nnet3${nnet3_affix}/ivectors_${train_id} || exit 1

  for d in dev; do 
    if [ ! -f exp/nnet3${nnet3_affix}/ivectors_$d/ivector_online.scp ]; then
      steps/online/nnet2/extract_ivectors_online.sh \
        --cmd "$train_cmd" --nj $nj \
        data/${d}_hires $extractor \
        exp/nnet3${nnet3_affix}/ivectors_${d} || exit 1
    fi
  done
fi

decode_opts="--extra-left-context 50 --extra-right-context 0 --extra-left-context-initial 0 --extra-right-context-final 0 --frames-per-chunk 150"
  
graph_affix=
graph_dir=$src_dir/graph${graph_affix}
lat_dir=$src_dir/decode${graph_affix}_${train_id}

if [ $stage -le 2 ]; then
  if [ ! -f $graph_dir/HCLG.fst ]; then
    utils/mkgraph.sh --self-loop-scale 1.0 --transition-scale 1.0 \
      $lang_test $src_dir $graph_dir
  fi
  
  steps/nnet3/decode_semisup.sh --nj $nj --cmd "$decode_cmd" --num-threads 4 \
    --acwt 1.0 --post-decode-acwt 10.0 --write-compact false \
    $decode_opts \
    --online-ivector-dir $train_ivector_dir \
    $graph_dir ${train_data_dir}_hires $lat_dir
fi

best_paths_dir=$src_dir/best_path${graph_affix}_$train_id
if [ $stage -le 3 ]; then
  steps/best_path_weights.sh --cmd "$train_cmd" --acwt 0.1 \
    $lat_dir $best_paths_dir
fi

if [ $stage -le 4 ]; then
  # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
  if [ ! -d "RIRS_NOISES" ]; then
    wget -O rirs_noises.zip --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
    rm rirs_noises.zip
  fi

  local/make_mx6.sh /export/corpora/LDC/LDC2013S03/mx6_speech data
  
  local/make_musan.sh /export/corpora/JHU/musan data

  for name in noise music; do
    utils/data/get_reco2dur.sh data/musan_${name}
  done

  utils/data/get_reco2dur.sh \
    --cmd "$train_cmd" --nj 4 data/mx6_mic

  utils/data/resample_data_dir.sh 8000 data/musan_noise
  utils/data/resample_data_dir.sh 8000 data/musan_music
  utils/data/resample_data_dir.sh 8000 data/mx6_mic
fi

rvb_data_dir=${train_data_dir}_rvb_hires

if [ $stage -le 5 ]; then
  rvb_opts=()
  # This is the config for the system using simulated RIRs and point-source noises
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  steps/data/augment_data_dir_for_asr.py --utt-prefix "noise" --fg-interval 1 \
    --fg-snrs "20:15:10:5:0" --fg-noise-dir "data/musan_noise" \
    ${train_data_dir}_hires ${train_data_dir}_noise || exit 1

  steps/data/augment_data_dir_for_asr.py --utt-prefix "music" \
    --bg-snrs "15:10:8:5" --num-bg-noises "1" \
    --bg-noise-dir "data/musan_music" \
    ${train_data_dir}_hires ${train_data_dir}_music || exit 1

  steps/data/augment_data_dir_for_asr.py --utt-prefix "babble" \
    --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" \
    --bg-noise-dir "data/mx6_mic" \
    ${train_data_dir}_hires ${train_data_dir}_babble || exit 1

  # corrupt the data to generate multi-condition data
  # for data_dir in train dev test; do
  seed=0
  for name in noise music babble; do 
    steps/data/reverberate_data_dir.py \
      "${rvb_opts[@]}" \
      --prefix "rev" \
      --speech-rvb-probability 0.8 \
      --pointsource-noise-addition-probability 0 \
      --isotropic-noise-addition-probability 0 \
      --num-replications 1 \
      --source-sampling-rate 8000 \
      --random-seed $seed \
      ${train_data_dir}_${name} ${train_data_dir}_${name}_reverb || exit 1
    seed=$[seed+1]
  done

  utils/combine_data.sh \
    ${rvb_data_dir} \
    ${train_data_dir}_noise_reverb \
    ${train_data_dir}_music_reverb \
    ${train_data_dir}_babble_reverb || exit 1
fi

if [ $stage -le 7 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $rvb_data_dir/data/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/fearless_steps-$(date +'%m_%d_%H_%M')/s5b/$rvb_data_dir/data/storage $rvb_data_dir/data/storage
  fi
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf \
    --cmd "$train_cmd --max-jobs-run 30" --nj $nj ${rvb_data_dir} || exit 1
  steps/compute_cmvn_stats.sh $rvb_data_dir
  utils/fix_data_dir.sh $rvb_data_dir
fi

rvb_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_id}_rvb

if [ $stage -le 8 ]; then
  steps/online/nnet2/extract_ivectors_online.sh \
    --cmd "$train_cmd" --nj $nj \
    ${rvb_data_dir} $extractor \
    exp/nnet3${nnet3_affix}/ivectors_${train_id}_rvb || exit 1
fi

rvb_lat_dir=$src_dir/decode${graph_affix}_${train_id}_rvb

if [ $stage -le 9 ]; then
  utt_prefixes=
  for n in noise music babble; do
    utt_prefixes="$utt_prefixes rev1-${n}_"
  done
  local/copy_lat_dir.sh \
    --cmd "$decode_cmd" --utt-prefixes "$utt_prefixes" \
    --nj $nj --write-compact false \
    ${rvb_data_dir} $lat_dir $rvb_lat_dir

  for n in noise music babble; do
    awk -v n=$n '{print "rev1-"n"_"$0}' $best_paths_dir/weights.scp
  done | sort -k1,1 > $rvb_lat_dir/weights.scp
fi

lang=data/lang_chain

if [ $stage -le 10 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt $src_lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r $src_lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

mkdir -p $dir

if [ $stage -le 11 ]; then
  echo "$0: compute {den,normalization}.fst using weighted phone LM."
  steps/nnet3/chain/make_weighted_den_fst.sh --cmd "$train_cmd" \
    --num-repeats $phone_lm_scales \
    --lm-opts '--num-extra-lm-states=2000' \
    $treedir $best_paths_dir $dir || exit 1;
fi

if [ $stage -le 12 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  tdnn_opts="l2-regularize=0.004"
  lstm_opts="l2-regularize=0.001 decay-time=20"
  output_opts="l2-regularize=0.002" 

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=$hidden_dim $tdnn_opts
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=$hidden_dim $tdnn_opts
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=$hidden_dim $tdnn_opts

  fast-lstmp-layer name=lstm1 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  fast-lstmp-layer name=lstm2 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  fast-lstmp-layer name=lstm3 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=tdnn8 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  relu-batchnorm-layer name=tdnn9 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  fast-lstmp-layer name=lstm4 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts

  ## adding the layers for chain branch
  output-layer name=output input=lstm4 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5 $output_opts

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=lstm4 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5 $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/fearless_steps-$(date +'%m_%d_%H_%M')/s5b/$dir/egs/storage $dir/egs/storage
  fi
  # exclude phone_LM and den.fst generation training stage
  if [ $train_stage -lt -4 ]; then train_stage=-4 ; fi

  # we use chain model from source to generate lats for target and the
  # tolerance used in chain egs generation using this lats should be 1 or 2 which is
  # (source_egs_tolerance/frame_subsampling_factor)
  # source_egs_tolerance = 5
  chain_opts=(--chain.alignment-subsampling-factor=1 --chain.left-tolerance=1 --chain.right-tolerance=1)

  steps/nnet3/chain/train.py --stage $train_stage ${chain_opts[@]} \
    --chain.mmi-factor-schedule="$mmi_factor_schedule" \
    --chain.kl-factor-schedule="$kl_factor_schedule" \
    --cmd="$train_cmd" \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.num-epochs $num_epochs \
    --feat.online-ivector-dir=$rvb_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --trainer.optimization.initial-effective-lrate=0.002 \
    --trainer.optimization.final-effective-lrate=0.0002 \
    --trainer.frames-per-iter=3000000 \
    --trainer.dropout-schedule="$dropout_schedule" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=true \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.optimization.momentum 0.0 \
    --trainer.deriv-truncate-margin 8 \
    --trainer.num-chunk-per-minibatch=64,32 \
    --egs.chunk-width $frames_per_eg \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --egs.opts="--frames-overlap-per-eg 0 --generate-egs-scp true --deriv-weights-scp $rvb_lat_dir/weights.scp --lattice-prune-beam $lattice_prune_beam --lattice-lm-scale $lattice_lm_scale --kl-latdir $rvb_lat_dir --kl-fst-scale $kl_fst_scale" \
    --egs.get-egs-script steps/nnet3/chain/get_egs_split.sh \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --feat-dir=${train_data_dir}_rvb_hires \
    --tree-dir=$treedir \
    --lat-dir=$rvb_lat_dir \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 15 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 $lang_test \
    $treedir $treedir/graph || exit 1;
fi

if [ $stage -le 16 ]; then
  frames_per_chunk=150
  rm $dir/.error 2>/dev/null || true

  for data in dev; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context 50 \
          --extra-right-context 0 \
          --extra-left-context-initial 0 --extra-right-context-final 0 \
          --frames-per-chunk 160 \
          --nj $nspk --cmd "$decode_cmd --max-jobs-run 64"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data} \
          $treedir/graph data/${data}_hires ${dir}/decode_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

#if [ $stage -le 16 ]; then
#   steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" \
#     $lang_test data/lang_pp_test_fg/ \
#     data/dev_hires ${dir}/decode_dev ${dir}/decode_dev.rescored
#fi

exit 0;
