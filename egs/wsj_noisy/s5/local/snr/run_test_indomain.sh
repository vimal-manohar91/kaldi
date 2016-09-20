#!/bin/bash

set -e 
set -o pipefail
set -u

. path.sh
. cmd.sh

feat_affix=_bp_vh
reco_nj=32

src_data_dir=data/dev10h.pem
data_dir=data/babel_assamese_dev10h
sad_nnet_dir=exp/nnet3_sad_snr/tdnn_irm_babel_assamese_train_unsad_splice5_2
sad_nnet_iter=final
create_uniform_segments=false
extra_left_context=0
overlap_length=500
window_length=3000
stage=-1

use_gpu=true

do_downsampling=false
segmentation_config=conf/segmentation.conf
weights_segmentation_config=conf/segmentation.conf
fbank_config=conf/fbank_bp.conf
mfcc_config=conf/mfcc_hires_bp.conf
pitch_config=conf/pitch.conf

echo $* 

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <src-data-dir> <data-dir> <sad-nnet-dir>"
  echo " e.g.: $0 $src_data_dir $data_dir $sad_nnet_dir"
  exit 1
fi

src_data_dir=$1
data_dir=$2
sad_nnet_dir=$3

data_id=`basename $data_dir`
sad_dir=${sad_nnet_dir}/sad_${data_id}_whole${feat_affix}
seg_dir=${sad_nnet_dir}/segmentation_${data_id}_whole${feat_affix}

export PATH="$KALDI_ROOT/tools/sph2pipe_v2.5/:$PATH"
[ ! -z `which sph2pipe` ]

if [ $stage -le 0 ]; then
  diarization/convert_data_dir_to_whole.sh $src_data_dir ${data_dir}_whole
  
  if $do_downsampling; then
    freq=`cat $mfcc_config | perl -pe 's/\s*#.*//g' | grep "sample-frequency=" | awk -F'=' '{if (NF == 0) print 16000; else print $2}'`
    sox=`which sox`

    cat $src_data_dir/wav.scp | python -c "import sys
for line in sys.stdin.readlines():
  splits = line.strip().split()
  if splits[-1] == '|':
    out_line = line.strip() + ' $sox -t wav - -r $freq -c 1 -b 16 -t wav - downsample |'
  else:
    out_line = 'cat {0} {1} | $sox -t wav - -r $freq -c 1 -b 16 -t wav - downsample |'.format(splits[0], ' '.join(splits[1:]))
  print (out_line)" > ${data_dir}_whole/wav.scp
  fi

  utils/copy_data_dir.sh ${data_dir}_whole ${data_dir}_whole${feat_affix}_hires
  utils/copy_data_dir.sh ${data_dir}_whole ${data_dir}_whole${feat_affix}_fbank
fi

if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --mfcc-config $mfcc_config --nj $reco_nj --cmd "$train_cmd" \
    ${data_dir}_whole${feat_affix}_hires exp/make_hires/${data_id}_whole${feat_affix} mfcc_hires
  steps/compute_cmvn_stats.sh ${data_dir}_whole${feat_affix}_hires exp/make_hires/${data_id}_whole${feat_affix} mfcc_hires
fi

#if [ $stage -le 2 ]; then
#  steps/make_fbank.sh --fbank-config $fbank_config --nj $reco_nj --cmd "$train_cmd" \
#    ${data_dir}_whole${feat_affix}_fbank exp/make_fbank/${data_id}_whole${feat_affix} fbank
#  steps/compute_cmvn_stats.sh ${data_dir}_whole${feat_affix}_fbank exp/make_fbank/${data_id}_whole${feat_affix} fbank
#fi

if [ $stage -le 5 ]; then
  local/snr/compute_sad.sh --nj $reco_nj --cmd "$train_cmd" --use-gpu $use_gpu --method Dnn \
    --iter $sad_nnet_iter --extra-left-context $extra_left_context --snr-data-dir ${data_dir}_whole${feat_affix}_hires \
    $sad_nnet_dir ${data_dir}_whole${feat_affix}_hires $sad_dir
fi

if ! $create_uniform_segments; then
  if [ $stage -le 6 ]; then
    local/snr/sad_to_segments.sh --config $segmentation_config --acwt 0.1 \
      --speech-to-sil-ratio 1.0 --sil-self-loop-probability 0.5 \
      --sil-transition-probability 0.5 ${data_dir}_whole${feat_affix}_fbank \
      $sad_dir $seg_dir $seg_dir/${data_id}_seg
  fi

  if [ $stage -le 7 ]; then
    local/snr/get_weights_for_ivector_extraction.sh --cmd queue.pl \
      --method Viterbi --config $weights_segmentation_config \
      --silence-weight 0 \
      ${seg_dir}/${data_id}_seg ${sad_dir} $seg_dir/ivector_weights_${data_id}_seg
  fi
else
  if [ $stage -le 6 ]; then
    local/snr/uniform_segment_data_dir.sh --overlap-length $overlap_length \
      --window-length $window_length \
      ${data_dir}_whole${feat_affix}_fbank $seg_dir $seg_dir/${data_id}_uniform_seg
  fi
  if [ $stage -le 7 ]; then
    local/snr/get_weights_for_ivector_extraction.sh --cmd queue.pl \
      --method Viterbi --config $weights_segmentation_config \
      --silence-weight 0 \
      ${seg_dir}/${data_id}_uniform_seg ${sad_dir} $seg_dir/ivector_weights_${data_id}_uniform_seg
  fi
fi

