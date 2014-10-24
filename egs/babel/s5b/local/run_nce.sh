#!/bin/bash

. conf/common_vars.sh
. ./lang.conf

set -e
set -o pipefail
set -u

num_iters=1
train_stage=-10
dir=dev10h.pem

. utils/parse_options.sh

if [ ! -f exp/tri5_nce/.done ]; then
  steps/train_nce.sh --stage $train_stage --cmd "$train_cmd" \
    --num-iters $num_iters \
    data/unsup.uem data/lang exp/tri5/decode_unsup.uem \
    exp/tri5_nce || exit 1
  touch exp/tri5_nce/.done
fi

dataset_dir=data/$dir
dataset_id=$dir
dataset_type=${dir%%.*}

eval my_nj=\$${dataset_type}_nj  #for shadow, this will be re-set when appropriate

decode=exp/tri5_nce/decode_${dataset_id}
if [ ! -f ${decode}/.done ]; then
  mkdir -p $decode
  steps/decode.sh --skip-scoring true \
    --nj $my_nj --cmd "$decode_cmd" "${decode_extra_opts[@]}" \
    --transform-dir exp/tri5/decode_${dataset_id} \
    exp/tri5/graph ${dataset_dir} ${decode} |tee ${decode}/decode.log
  touch ${decode}/.done
fi
  
local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
  --skip-scoring $skip_scoring --extra-kws $extra_kws --wip $wip \
  --cmd "$decode_cmd" --skip-kws $skip_kws --skip-stt $skip_stt \
  "${lmwt_plp_extra_opts[@]}" \
  ${dataset_dir} data/lang ${decode}
