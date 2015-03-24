#!/bin/bash

. conf/common_vars.sh
. ./lang.conf

set -e
set -o pipefail
set -u

degs_dir=
criterion=smbr
dev_dir=data/dev2h.pem
one_silence_class=true
src_dir=exp/tri6_nnet
graph_dir=exp/tri5/graph
dir=exp/tri6_nnet_smbr

. utils/parse_options.sh

if ! $one_silence_class; then
  dir=${dir}_noonesilence
fi

# Wait for cross-entropy training.
echo "Waiting till ${src_dir}/.done exists...."
while [ ! -f ${src_dir}/.done ]; do sleep 30; done
echo "...done waiting for ${src_dir}/.done"

# Generate denominator lattices.
if [ ! -f ${src_dir}_denlats/.done ]; then
  steps/nnet2/make_denlats.sh "${dnn_denlats_extra_opts[@]}" \
    --nj $train_nj --sub-split $train_nj \
    --transform-dir exp/tri5_ali \
    data/train data/lang ${src_dir} ${src_dir}_denlats || exit 1
 
  touch ${src_dir}_denlats/.done
fi

# Generate alignment.
if [ ! -f ${src_dir}_ali/.done ]; then
  steps/nnet2/align.sh --use-gpu yes \
    --cmd "$decode_cmd $dnn_parallel_opts" \
    --transform-dir exp/tri5_ali --nj $train_nj \
    data/train data/lang ${src_dir} ${src_dir}_ali || exit 1

  touch ${src_dir}_ali/.done
fi

lang=`echo $train_data_dir | perl -pe 's:.+data/\d+-([^/]+)/.+:$1:'`

if [[ `hostname -f` == *.clsp.jhu.edu ]]; then
  # spread the egs over various machines. 
  [ -z "$degs_dir" ] && utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/babel_${lang}_s5b-$(date +'%d_%m_%H_%M')/$dir/degs $dir/degs/storage 
fi

if [ -z "$degs_dir" ]; then
  if [ ! -f $dir/.degs.done ]; then
    steps/nnet2/get_egs_discriminative2.sh --cmd "$decode_cmd --max-jobs-run 10" \
      --criterion $criterion --drop-frames true \
      --transform-dir exp/tri5_ali \
      data/train data/lang \
      ${src_dir}_ali ${src_dir}_denlats \
      ${src_dir}/final.mdl $dir/degs || exit 1
    touch $dir/.degs.done
  fi
  degs_dir=$dir/degs
fi

train_stage=-100
if [ ! -f $dir/.done ]; then
  steps/nnet2/train_discriminative2.sh \
    --criterion $criterion \
    --stage $train_stage --cmd "$decode_cmd --num-threads 16" \
    --learning-rate $dnn_mpe_learning_rate \
    --modify-learning-rates true \
    --last-layer-factor $dnn_mpe_last_layer_factor \
    --num-epochs 4 --cleanup false \
    --retroactive $dnn_mpe_retroactive --one-silence-class $one_silence_class \
    "${dnn_cpu_mpe_parallel_opts[@]}" \
    $degs_dir $dir || exit 1

  touch $dir/.done
fi

dev_id=$(basename $dev_dir)
eval my_nj=\$${dev_id%%.*}_nj

if [ -f $dir/.done ]; then
  for epoch in 3 4; do
    decode=$dir/decode_${dev_id}_epoch$epoch
    if [ ! -f $decode/.done ]; then
      mkdir -p $decode
      steps/nnet2/decode.sh --minimize $minimize \
        --cmd "$decode_cmd --mem 4G --num-threads 6" --nj $my_nj --iter epoch$epoch \
        --beam $dnn_beam --lattice-beam $dnn_lat_beam \
        --skip-scoring true --num-threads 6 \
        --transform-dir exp/tri5/decode_${dev_id} \
        $graph_dir ${dev_dir} $decode | tee $decode/decode.log

      touch $decode/.done
    fi

    local/run_kws_stt_task.sh --cer $cer --max-states 150000 \
      --skip-scoring false --extra-kws false --wip 0.5 \
      --cmd "$decode_cmd" --skip-kws true --skip-stt false \
      "${lmwt_dnn_extra_opts[@]}" \
      ${dev_dir} data/lang $decode
  done
fi

