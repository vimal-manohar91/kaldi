#!/bin/bash

set -e 
set -o pipefail
set -u

. path.sh
. cmd.sh

feat_affix=bp_vh
affix=
reco_nj=32

stage=-1
sad_stage=-1

# SAD network config
sad_nnet_iter=final

extra_left_context=0            # Set to some large value, typically 40 for LSTM (must match training)

# Use gpu for nnet propagation
use_gpu=true

# Set to true if the test data has > 8kHz sampling frequency.
do_downsampling=false

# Configs
segmentation_config=conf/segmentation_speech.conf
mfcc_config=conf/mfcc_hires_bp.conf

echo $* 

. utils/parse_options.sh

src_data_dir=data/dev10h.pem
data_dir=data/babel_assamese_dev10h
sad_nnet_dir=exp/nnet3_sad_snr/tdnn_a_n4

if [ $# -ne 3 ]; then
  echo "Usage: $0 <src-data-dir> <data-dir> <sad-nnet-dir>"
  echo " e.g.: $0 $src_data_dir $data_dir $sad_nnet_dir"
  exit 1
fi

src_data_dir=$1
data_dir=$2
sad_nnet_dir=$3

affix=${affix:+_$affix}
feat_affix=${feat_affix:+_$feat_affix}

data_id=`basename $data_dir`
sad_dir=${sad_nnet_dir}/sad${affix}_${data_id}_whole${feat_affix}
seg_dir=${sad_nnet_dir}/segmentation${affix}_${data_id}_whole${feat_affix}

export PATH="$KALDI_ROOT/tools/sph2pipe_v2.5/:$PATH"
[ ! -z `which sph2pipe` ]

if [ $stage -le 0 ]; then
  utils/data/convert_data_dir_to_whole.sh $src_data_dir ${data_dir}_whole
  
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
fi

test_data_dir=${data_dir}_whole${feat_affix}_hires

if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --mfcc-config $mfcc_config --nj $reco_nj --cmd "$train_cmd" \
    ${data_dir}_whole${feat_affix}_hires exp/make_hires/${data_id}_whole${feat_affix} mfcc_hires
  steps/compute_cmvn_stats.sh ${data_dir}_whole${feat_affix}_hires exp/make_hires/${data_id}_whole${feat_affix} mfcc_hires
fi

if [ $stage -le 5 ]; then
  local/segmentation/compute_sad.sh --nj $reco_nj --cmd "$train_cmd" --use-gpu $use_gpu \
    --iter $sad_nnet_iter --extra-left-context $extra_left_context --stage $sad_stage \
    ${test_data_dir} $sad_nnet_dir $sad_dir
fi

if [ $stage -le 6 ]; then
  steps/segmentation/post_process_sad_to_segments.sh \
    --segmentation-config $segmentation_config --nj $reco_nj --cmd "$train_cmd" \
    ${test_data_dir} $sad_dir/speech_prob.scp $seg_dir $seg_dir/${data_id}_seg
    echo "$0: Created segmented data dir in $seg_dir/${data_id}_seg"
fi
