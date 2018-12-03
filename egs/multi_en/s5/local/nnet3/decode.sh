#!/bin/bash

decode_iter=
decode_dir_affix=
decode_nj=50
test_sets="rt03"
stage=0

# decode options
frames_per_chunk_primary=140
extra_left_context=50
extra_right_context=0

. ./cmd.sh
. ./utils/parse_options.sh

set -e -o pipefail -u

if [ $# -ne 4 ]; then
  exit 1
fi

lang_dir=$1
rescore_lang_dir=$2
ivector_root_dir=$3
dir=$4

lang_suffix=${lang_dir##*lang}

if [ $stage -le 1 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 $lang_dir \
    $dir $dir/graph${lang_suffix}
fi

graph_dir=$dir/graph${lang_suffix}
if [ $stage -le 2 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in $test_sets; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
        --nj 50 --cmd "$decode_cmd" $iter_opts \
        --extra-left-context $extra_left_context \
        --extra-right-context $extra_right_context \
        --extra-left-context-initial 0 \
        --extra-right-context-final 0 \
        --frames-per-chunk "$frames_per_chunk_primary" \
        --online-ivector-dir $ivector_root_dir/ivectors_${decode_set} \
        $graph_dir data/${decode_set}_hires \
         $dir/decode${lang_suffix}_${decode_set}${decode_dir_affix:+_$decode_dir_affix} 
      
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
          $lang_dir $rescore_lang_dir data/${decode_set}_hires \
          $dir/decode${lang_suffix}_${decode_set}${decode_dir_affix:+_$decode_dir_affix}{,_fg} || exit 1;
      ) &
  done
fi
wait;

