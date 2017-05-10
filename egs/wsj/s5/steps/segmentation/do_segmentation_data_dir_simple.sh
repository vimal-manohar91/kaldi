#!/bin/bash

set -e 
set -o pipefail
set -u

. path.sh
. cmd.sh

affix=  # Affix for the segmentation
nj=32  # works on recordings as against on speakers

# Feature options (Must match training)
mfcc_config=conf/mfcc_hires_bp.conf
feat_affix=bp   # Affix for the type of feature used

convert_data_dir_to_whole=true

# Set to true if the test data has > 8kHz sampling frequency.
do_downsampling=false

stage=-1
sad_stage=-1
output_name=output-speech   # The output node in the network
sad_name=sad    # Base name for the directory storing the computed loglikes
segmentation_name=segmentation  # Base name for the directory doing segmentation

# SAD network config
iter=final  # Model iteration to use

# Contexts must ideally match training for LSTM models, but
# may not necessarily for stats components
extra_left_context=0  # Set to some large value, typically 40 for LSTM (must match training)
extra_right_context=0  

frame_subsampling_factor=1  # Subsampling at the output

transition_scale=3.0
loopscale=0.1
acwt=1.0

# Segmentation configs
segmentation_config=conf/segmentation_speech.conf

echo $* 

. utils/parse_options.sh

if [ $# -ne 5 ]; then
  echo "Usage: $0 <src-data-dir> <sad-nnet-dir> <lang> <mfcc-dir> <data-dir>"
  echo " e.g.: $0 ~/workspace/egs/ami/s5b/data/sdm1/dev exp/nnet3_sad_snr/nnet_tdnn_j_n4 mfcc_hires_bp data/ami_sdm1_dev"
  exit 1
fi

src_data_dir=$1   # The input data directory that needs to be segmented.
                  # Any segments in that will be ignored.
sad_nnet_dir=$2   # The SAD neural network
lang=$3
mfcc_dir=$4       # The directory to store the features
data_dir=$5       # The output data directory will be ${data_dir}_seg

affix=${affix:+_$affix}
feat_affix=${feat_affix:+_$feat_affix}

data_id=`basename $data_dir`
sad_dir=${sad_nnet_dir}/${sad_name}${affix}_${data_id}_whole${feat_affix}
seg_dir=${sad_nnet_dir}/${segmentation_name}${affix}_${data_id}_whole${feat_affix}

export PATH="$KALDI_ROOT/tools/sph2pipe_v2.5/:$PATH"
[ ! -z `which sph2pipe` ]

test_data_dir=data/${data_id}${feat_affix}_hires

if $convert_data_dir_to_whole; then
  if [ $stage -le 0 ]; then
    whole_data_dir=${sad_dir}/${data_id}_whole
    utils/data/convert_data_dir_to_whole.sh $src_data_dir ${whole_data_dir}
    
    if $do_downsampling; then
      freq=`cat $mfcc_config | perl -pe 's/\s*#.*//g' | grep "sample-frequency=" | awk -F'=' '{if (NF == 0) print 16000; else print $2}'`
      utils/data/downsample_data_dir.sh $freq $whole_data_dir
    fi

    rm -r ${test_data_dir} || true
    utils/copy_data_dir.sh ${whole_data_dir} $test_data_dir
  fi
else
  if [ $stage -le 0 ]; then
    rm -r ${test_data_dir} || true
    utils/copy_data_dir.sh $src_data_dir $test_data_dir

    if $do_downsampling; then
      freq=`cat $mfcc_config | perl -pe 's/\s*#.*//g' | grep "sample-frequency=" | awk -F'=' '{if (NF == 0) print 16000; else print $2}'`
      utils/data/downsample_data_dir.sh $freq $test_data_dir
    fi
  fi
fi
  
if [ $stage -le 1 ]; then
  utils/fix_data_dir.sh $test_data_dir
  steps/make_mfcc.sh --mfcc-config $mfcc_config --nj $nj --cmd "$train_cmd" \
    ${test_data_dir} exp/make_hires/${data_id}${feat_affix} $mfcc_dir
  steps/compute_cmvn_stats.sh ${test_data_dir} exp/make_hires/${data_id}${feat_affix} $mfcc_dir
  utils/fix_data_dir.sh ${test_data_dir}
fi

post_vec=$sad_nnet_dir/post_${output_name}.vec
if [ ! -f $sad_nnet_dir/post_${output_name}.vec ]; then
  echo "$0: Could not find $sad_nnet_dir/post_${output_name}.vec. See the last stage of local/segmentation/run_train_sad.sh"
  exit 1
fi

create_topo=true
if $create_topo; then
  if [ ! -f $lang/classes_info.txt ]; then
    echo "$0: Could not find $lang/topo or $lang/classes_info.txt"
    exit 1
  else
    steps/segmentation/internal/prepare_simple_hmm_lang.py \
      $lang/classes_info.txt $lang
  fi
fi

if [ $stage -le 3 ]; then
  simple-hmm-init $lang/topo $lang/init.mdl 

  $train_cmd $sad_nnet_dir/log/get_final_${output_name}_model.log \
    nnet3-am-init $lang/init.mdl \
    "nnet3-copy --edits='rename-node old-name=$output_name new-name=output' $sad_nnet_dir/$iter.raw - |" - \| \
    nnet3-am-adjust-priors - $sad_nnet_dir/post_${output_name}.vec \
    $sad_nnet_dir/${iter}_${output_name}.mdl
fi
iter=${iter}_${output_name}

if [ $stage -le 4 ]; then
  steps/nnet3/compute_output.sh --nj $nj --cmd "$train_cmd" \
    --iter $iter --use-raw-nnet false \
    --extra-left-context $extra_left_context \
    --extra-right-context $extra_right_context \
    --frames-per-chunk 150 \
    --stage $sad_stage \
    --frame-subsampling-factor $frame_subsampling_factor \
    ${test_data_dir} $sad_nnet_dir $sad_dir
fi

graph_dir=${sad_nnet_dir}/graph_${output_name}

if [ $stage -le 5 ]; then
  cp -r $lang $graph_dir

  if [ ! -f $lang/final.mdl ]; then
    echo "$0: Could not find $lang/final.mdl!"
    echo "$0: Using $lang/init.mdl instead"
    cp $lang/init.mdl $graph_dir/final.mdl
  else
    cp $lang/final.mdl $graph_dir
  fi

  $train_cmd $lang/log/make_graph.log \
    make-simple-hmm-graph --transition-scale=$transition_scale \
    --self-loop-scale=$loopscale \
    $graph_dir/final.mdl \| \
    fstdeterminizestar --use-log=true \| \
    fstrmepslocal \| \
    fstminimizeencoded '>' $graph_dir/HCLG.fst
fi

if [ $stage -le 6 ]; then
  steps/segmentation/decode_sad.sh --acwt 1.0 --cmd "$decode_cmd" \
    --iter ${iter} \
    --get-pdfs true $graph_dir $sad_dir $seg_dir
fi

if [ $stage -le 7 ]; then
  steps/segmentation/post_process_sad_to_subsegments.sh \
    --cmd "$train_cmd" --segmentation-config $segmentation_config \
    --frame-subsampling-factor $frame_subsampling_factor \
    ${test_data_dir} $lang/phone2sad_map ${seg_dir} \
    ${seg_dir} ${data_dir}_seg

  cp $src_data_dir/wav.scp ${data_dir}_seg
fi
