#!/bin/bash

. conf/common_vars.sh
. ./lang.conf

set -e
set -o pipefail
set -u

degs_dir=
uegs_dir=
egs_dir=
best_path_dir=exp/tri6_nnet/best_path_unsup.uem
criterion=smbr
num_jobs_nnet="6 2"
learning_rate_scales="1.0 2.0"
last_layer_factor="0.1 0.1"
dir=exp/tri6_nnet_nce
dev_dir=data/dev2h.pem
train_stage=-100
lm_order=3
boost=0.1
nce_boost=0.0
skip_last_layer=true
reduce_scale_factor=
one_silence_class=true
weight_threshold=0.0
src_models=

. utils/parse_options.sh
    
dir=${dir}_${criterion}_supscale$(echo $learning_rate_scales | awk '{printf $2}')
if [ ! -z "$reduce_scale_factor" ]; then
  dir=${dir}_reduce_scale_factor${reduce_scale_factor}
fi

dir=${dir}_lr_${dnn_mpe_learning_rate}_nj$(echo $num_jobs_nnet | sed 's/ /_/g')

if [ $(echo $last_layer_factor | awk '{printf $2}') != 0.1 ]; then
  dir=${dir}_llf$(echo $last_layer_factor | sed 's/ /_/g')
fi

if [ "$lm_order" -eq 1 ]; then
  dir=${dir}_lmorder${lm_order}
fi

if [ "$boost" != 0.1 ]; then
  dir=${dir}_bmmi${boost}
fi

if [ "$nce_boost" != 0.0 ]; then
  dir=${dir}_bnce${nce_boost}
fi

if [ "$weight_threshold" != 0.0 ]; then
  dir=${dir}_thres${weight_threshold}
fi

if ! $skip_last_layer; then
  dir=${dir}_noskip
fi

if $one_silence_class; then
  dir=${dir}_onesilence
fi

# Wait for cross-entropy training.
echo "Waiting till exp/tri6_nnet/.done exists...."
while [ ! -f exp/tri6_nnet/.done ]; do sleep 30; done
echo "...done waiting for exp/tri6_nnet/.done"

# Generate denominator lattices.
if [ ! -f exp/tri6_nnet_denlats/.done ]; then
  steps/nnet2/make_denlats.sh "${dnn_denlats_extra_opts[@]}" \
    --nj $train_nj --sub-split $train_nj \
    --transform-dir exp/tri5_ali \
    data/train data/lang exp/tri6_nnet exp/tri6_nnet_denlats || exit 1
 
  touch exp/tri6_nnet_denlats/.done
fi

# Generate alignment.
if [ ! -f exp/tri6_nnet_ali/.done ]; then
  steps/nnet2/align.sh --use-gpu no \
    --cmd "$decode_cmd" \
    --transform-dir exp/tri5_ali --nj $train_nj \
    data/train data/lang exp/tri6_nnet exp/tri6_nnet_ali || exit 1

  touch exp/tri6_nnet_ali/.done
fi

if [ $lm_order -ne 1 ] && [ ! -f exp/tri6_nnet/decode_unsup.uem/.done ]; then
  echo "$0: Expecting unlabelled data to be decoded in exp/tri6_nnet/decode_unsup.uem. Run run-4-anydecode.sh --dir unsup.uem first."
  exit 1
fi

lang=`echo $train_data_dir | perl -pe 's:.+data/\d+-([^/]+)/.+:$1:'`

if [[ `hostname -f` == *.clsp.jhu.edu ]]; then
  # spread the egs over various machines. 
  [ -z "$degs_dir" ] && utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/babel_${lang}_s5b-$(date +'%d_%m_%H_%M')/$dir/degs $dir/degs/storage 

  [ -z "$uegs_dir" ] && utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/babel_${lang}_s5b-$(date +'%d_%m_%H_%M')/$dir/uegs $dir/uegs/storage 
fi

if [ -z "$degs_dir" ]; then
  if [ ! -f $dir/.degs.done ]; then
    steps/nnet2/get_egs_discriminative2.sh --cmd "$decode_cmd --max-jobs-run 10" \
      --criterion $criterion --drop-frames true \
      --transform-dir exp/tri5_ali \
      data/train data/lang \
      exp/tri6_nnet_ali exp/tri6_nnet_denlats \
      exp/tri6_nnet/final.mdl $dir/degs || exit 1
    touch $dir/.degs.done
  fi
  degs_dir=$dir/degs
fi

if [ -z "$uegs_dir" ]; then
  unsup_decode_dir=exp/tri6_nnet/decode_unsup.uem
  [ "$lm_order" -eq 1 ] && unsup_decode_dir=exp/tri6_nnet/decode_lmorder${lm_order}_unsup.uem

  if [ "$lm_order" -eq 1 ] &&  [ ! -f exp/tri6_nnet/decode_lmorder${lm_order}_unsup.uem/.done ]; then 
    steps/nnet2/make_denlats.sh "${dnn_denlats_extra_opts[@]}" \
      --nj $unsup_nj --sub-split $unsup_nj \
      --transform-dir exp/tri5/decode_unsup.uem \
      --text data/train/text \
      --lattice-beam 8.0 --beam 15.0 --max-active 7000 --min-active 200 \
      data/unsup.uem data/lang exp/tri6_nnet exp/tri6_nnet/decode_lmorder${lm_order}_unsup.uem || exit 1

    touch exp/tri6_nnet/decode_lmorder${lm_order}_unsup.uem/.done
  fi

  if [ ! -f $best_path_dir/.done ]; then
    local/best_path_weights.sh --cmd "$decode_cmd" --create-ali-dir true \
      data/unsup.uem exp/tri5/graph exp/tri6_nnet/decode_unsup.uem $best_path_dir || exit 1
    touch $best_path_dir/.done
  fi

  if [ ! -f $dir/.uegs.done ]; then
    steps/nnet2/get_uegs2.sh --cmd "$decode_cmd --max-jobs-run 10" \
      --transform-dir exp/tri5/decode_unsup.uem \
      --alidir $best_path_dir \
      data/unsup.uem data/lang \
      $unsup_decode_dir \
      exp/tri6_nnet/final.mdl $dir/uegs || exit 1
    touch $dir/.uegs.done
  fi
  uegs_dir=$dir/uegs
fi

if [ ! -f $dir/.done ]; then
  steps/nnet2/train_discriminative_semisupervised_multinnet2.sh \
    --criterion $criterion \
    --stage $train_stage --cmd "$train_cmd" \
    --learning-rate $dnn_mpe_learning_rate \
    --separate-learning-rates true \
    --modify-learning-rates true \
    --learning-rate-scales "$learning_rate_scales" --reduce-scale-factor "$reduce_scale_factor" \
    --last-layer-factor "$last_layer_factor" \
    --num-epochs 4 --cleanup false --weight-threshold $weight_threshold \
    --boost $boost --nce-boost $nce_boost --one-silence-class $one_silence_class \
    --retroactive $dnn_mpe_retroactive --num-threads 16 \
    --num-jobs-nnet "$num_jobs_nnet" --skip-last-layer $skip_last_layer \
    --egs-dir "$egs_dir" --src-models "$src_models" \
    $uegs_dir $degs_dir $dir || exit 1

  touch $dir/.done
fi

dev_id=$(basename $dev_dir)
eval my_nj=\$${dev_id%%.*}_nj

if [ -f $dir/.done ]; then
  for lang in 1 0; do
    for epoch in 3 4; do
      decode=$dir/$lang/decode_${dev_id}_epoch$epoch
      if [ ! -f $decode/.done ]; then
        mkdir -p $decode
        steps/nnet2/decode.sh --minimize $minimize \
          --cmd "$decode_cmd" --parallel-opts "--mem 4G --num-threads 6" --nj $my_nj --iter epoch$epoch \
          --beam $dnn_beam --lattice-beam $dnn_lat_beam \
          --skip-scoring true --num-threads 6 \
          --transform-dir exp/tri5/decode_${dev_id} \
          exp/tri5/graph ${dev_dir} $decode | tee $decode/decode.log

        touch $decode/.done
      fi

      local/run_kws_stt_task.sh --cer $cer --max-states 150000 \
        --skip-scoring false --extra-kws false --wip 0.5 \
        --cmd "$decode_cmd" --skip-kws true --skip-stt false \
        "${lmwt_dnn_extra_opts[@]}" \
        ${dev_dir} data/lang $decode
    done
  done
fi

