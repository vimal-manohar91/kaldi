#!/bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script prepares speech labels and deriv weights for 
# training unsad network for speech activity detection and music detection.

set -u
set -o pipefail
set -e

. path.sh

stage=-3
cmd=queue.pl
reco_nj=40      # Number of jobs to work at recording-level
nj=100          # Number of jobs to work at utterance-level

# Options to be passed to get_sad_map.py 
map_noise_to_sil=true   # Map noise phones to silence label (0)
map_unk_to_speech=true  # Map unk phones to speech label (1)
sad_map=    # Initial mapping from phones to speech/non-speech labels.
            # Overrides the default mapping using phones/silence.txt 
            # and phones/nonsilence.txt
            # The format is <phone> <label>

# Options for feature extraction 
# (These must match the features used for model_dir and sat_model_dir)
feat_type=mfcc        # mfcc or plp
add_pitch=false       # Add pitch features

config_dir=conf
feat_config=
pitch_config=     

mfccdir=mfcc
plpdir=plp

speed_perturb=true

sat_model_dir=  # Model directory used for getting alignments
lang_test=  # Language directory used to build graph. 
            # If its not provided, $lang will be used instead.

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "This script takes a data directory and creates a new data directory "
  echo "and speech activity labels"
  echo "for the purpose of training a Universal Speech Activity Detector."
  echo "Usage: $0 [options] <data-dir> <lang> <model-dir> <temp-dir>"
  echo " e.g.: $0 data/train_100k data/lang exp/tri4a exp/vad_data_prep"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (run.pl|/queue.pl <queue opts>)            # how to run jobs."
  echo "  --reco-nj <#njobs|4>                             # Split a whole data directory into these many pieces"
  echo "  --nj      <#njobs|4>                             # Split a segmented data directory into these many pieces"
  exit 1
fi

data_dir=$1
lang=$2
model_dir=$3
dir=$4

if [ $feat_type != "plp" ] && [ $feat_type != "mfcc" ]; then
  echo "$0: --feat-type must be plp or mfcc. Must match the model_dir used."
  exit 1
fi

[ -z "$feat_config" ] && feat_config=$config_dir/$feat_type.conf
[ -z "$pitch_config" ] && pitch_config=$config_dir/pitch.conf

extra_files=

if $add_pitch; then
  extra_files="$extra_files $pitch_config"
fi

for f in $feat_config $extra_files; do
  if [ ! -f $f ]; then
    echo "$f could not be found"
    exit 1
  fi
done

mkdir -p $dir

function make_mfcc {
  local nj=$nj
  local mfcc_config=$feat_config
  local add_pitch=$add_pitch
  local cmd=$cmd
  local pitch_config=$pitch_config

  while [ $# -gt 0 ]; do 
    if [ $1 == "--nj" ]; then
      nj=$2
      shift; shift;
    elif [ $1 == "--mfcc-config" ]; then
      mfcc_config=$2
      shift; shift;
    elif [ $1 == "--add-pitch" ]; then
      add_pitch=$2
      shift; shift;
    elif [ $1 == "--cmd" ]; then
      cmd=$2
      shift; shift;
    elif [ $1 == "--pitch-config" ]; then
      pitch_config=$2
      shift; shift;
    else
      break
    fi
  done

  if [ $# -ne 3 ]; then
    echo "Usage: make_mfcc <data-dir> <temp-dir> <feat-dir>"
    exit 1
  fi

  if $add_pitch; then
    steps/make_mfcc_pitch.sh --cmd "$cmd" --nj $nj --write-utt2num-frames true \
      --mfcc-config $mfcc_config --pitch-config $pitch_config $1 $2 $3 || exit 1
  else
    steps/make_mfcc.sh --cmd "$cmd" --nj $nj --write-utt2num-frames true \
      --mfcc-config $mfcc_config $1 $2 $3 || exit 1
  fi

}

function make_plp {
  local nj=$nj
  local mfcc_config=$feat_config
  local add_pitch=$add_pitch
  local cmd=$cmd
  local pitch_config=$pitch_config
  
  while [ $# -gt 0 ]; do 
    if [ $1 == "--nj" ]; then
      nj=$2
      shift; shift;
    elif [ $1 == "--plp-config" ]; then
      plp_config=$2
      shift; shift;
    elif [ $1 == "--add-pitch" ]; then
      add_pitch=$2
      shift; shift;
    elif [ $1 == "--cmd" ]; then
      cmd=$2
      shift; shift;
    elif [ $1 == "--pitch-config" ]; then
      pitch_config=$2
      shift; shift;
    else
      break
    fi
  done

  if [ $# -ne 3 ]; then
    echo "Usage: make_plp <data-dir> <temp-dir> <feat-dir>"
    exit 1
  fi
  
  if $add_pitch; then
    steps/make_plp_pitch.sh --cmd "$cmd" --nj $nj --write-utt2num-frames true \
      --plp-config $plp_config --pitch-config $pitch_config $1 $2 $3 || exit 1
  else
    steps/make_plp.sh --cmd "$cmd" --nj $nj --write-utt2num-frames true \
      --plp-config $plp_config $1 $2 $3 || exit 1
  fi
}

frame_shift_info=`cat $feat_config | steps/segmentation/get_frame_shift_info_from_config.pl` || exit 1

frame_shift=`echo $frame_shift_info | awk '{print $1}'`
frame_overlap=`echo $frame_shift_info | awk '{print $2}'`
  
data_id=$(basename $data_dir)
whole_data_dir=${data_dir}_whole
whole_data_id=${data_id}_whole

if [ $stage -le -1 ]; then
  utils/data/convert_data_dir_to_whole.sh ${data_dir} ${whole_data_dir}
fi 

if $speed_perturb; then
  plpdir=${plpdir}_sp
  mfccdir=${mfccdir}_sp
 
  if [ $stage -le 0 ]; then
    utils/data/perturb_data_dir_speed_3way.sh ${whole_data_dir} ${whole_data_dir}_sp
    utils/data/perturb_data_dir_speed_3way.sh ${data_dir} ${data_dir}_sp

    if [ $feat_type == "mfcc" ]; then
      if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
        utils/create_split_dir.pl \
          /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
      fi
      make_mfcc --cmd "$cmd --max-jobs-run 40" --nj $nj \
        --mfcc-config $feat_config \
        --add-pitch $add_pitch --pitch-config $pitch_config \
        ${whole_data_dir}_sp exp/make_mfcc $mfccdir || exit 1
      steps/compute_cmvn_stats.sh \
        ${whole_data_dir}_sp exp/make_mfcc $mfccdir || exit 1
    elif [ $feat_type == "plp" ]; then
      if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $plpdir/storage ]; then
        utils/create_split_dir.pl \
          /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$plpdir/storage $plpdir/storage
      fi

      make_plp --cmd "$cmd --max-jobs-run 40" --nj $nj \
        --plp-config $feat_config \
        --add-pitch $add_pitch --pitch-config $pitch_config \
        ${whole_data_dir}_sp exp/make_plp $plpdir || exit 1
      steps/compute_cmvn_stats.sh \
        ${whole_data_dir}_sp exp/make_plp $plpdir || exit 1
    else
      echo "$0: Unknown feat-type $feat_type. Must be mfcc or plp."
      exit 1
    fi
        
    utils/fix_data_dir.sh ${whole_data_dir}_sp
  fi

  data_dir=${data_dir}_sp
  whole_data_dir=${whole_data_dir}_sp
  data_id=${data_id}_sp
fi

if [ $stage -le 1 ]; then
  # Get feats for the manual segments
  utils/data/subsegment_data_dir.sh $whole_data_dir ${data_dir}/segments ${data_dir}/tmp
  cp $data_dir/tmp/feats.scp $data_dir

  if [ $feat_type == mfcc ]; then
    steps/compute_cmvn_stats.sh ${data_dir} exp/make_mfcc/${data_id} $mfccdir
  else
    steps/compute_cmvn_stats.sh ${data_dir} exp/make_plp/${data_id} $plpdir
  fi

  utils/fix_data_dir.sh $data_dir
fi

#obtain the alignment of the perturbed data
if [ -z "$sat_model_dir" ]; then
  ali_dir=$dir/`basename ${model_dir}`_ali_${data_id}
  if [ $stage -le 2 ]; then
    steps/align_si.sh --nj $nj --cmd "$cmd" \
      ${data_dir} ${lang} ${model_dir} $ali_dir || exit 1
  fi
else
  ali_dir=$dir/`basename ${sat_model_dir}`_ali_${data_id}
  if [ $stage -le 2 ]; then
    steps/align_fmllr.sh --nj $nj --cmd "$cmd" \
      ${data_dir} ${lang} ${sat_model_dir} $ali_dir || exit 1
  fi
fi


# All the data from this point is speed perturbed.

data_id=$(basename $data_dir)
utils/split_data.sh $data_dir $nj

###############################################################################
# Convert alignment for the provided segments into 
# initial SAD labels at utterance-level in segmentation format
###############################################################################

vad_dir=$dir/`basename ${ali_dir}`_vad_${data_id}
if [ $stage -le 3 ]; then
  steps/segmentation/get_sad_map.py \
    --init-sad-map="$sad_map" \
    --map-noise-to-sil=$map_noise_to_sil \
    --map-unk-to-speech=$map_unk_to_speech \
    $lang | utils/sym2int.pl -f 1 $lang/phones.txt > $dir/sad_map

  steps/segmentation/internal/convert_ali_to_vad.sh --cmd "$cmd" \
    $ali_dir $dir/sad_map $vad_dir
fi

[ ! -s $vad_dir/sad_seg.scp ] && echo "$0: $vad_dir/vad.scp is empty" && exit 1

if [ $stage -le 4 ]; then
  utils/copy_data_dir.sh $data_dir $dir/${data_id}_manual_segments

  awk '{print $1" "$2}' $dir/${data_id}_manual_segments/segments | sort -k1,1 > $dir/${data_id}_manual_segments/utt2spk
  utils/utt2spk_to_spk2utt.pl $dir/${data_id}_manual_segments/utt2spk | sort -k1,1 > $dir/${data_id}_manual_segments/spk2utt

  if [ $feat_type == mfcc ]; then
    steps/compute_cmvn_stats.sh $dir/${data_id}_manual_segments exp/make_mfcc/${data_id}_manual_segments $mfccdir
  else
    steps/compute_cmvn_stats.sh $dir/${data_id}_manual_segments exp/make_plp/${data_id}_manual_segments $plpdir
  fi
  
  utils/fix_data_dir.sh $dir/${data_id}_manual_segments || true     # Might fail because utt2spk will be not sorted on both utts and spks
fi

  
# Create extended data directory that consists of the provided 
# segments along with the segments outside it.
# This is basically dividing the whole recording into pieces
# consisting of pieces corresponding to the provided segments 
# and outside the provided segments.

###############################################################################
# Create segments outside of the manual segments and split them 
# into 10s long chunks.
###############################################################################

outside_data_dir=$dir/${data_id}_outside
if [ $stage -le 5 ]; then
  rm -r $outside_data_dir || true
  mkdir -p $outside_data_dir

  for f in wav.scp reco2file_and_channel stm glm; do 
    if [ -f ${data_dir}/$f ]; then
      cp ${data_dir}/$f $outside_data_dir
    fi
  done
   
  utils/data/get_utt2num_frames.sh $whole_data_dir

  $cmd $outside_data_dir/log/get_outside_segments.log \
    segmentation-init-from-segments --frame-shift=$frame_shift \
      --frame-overlap=$frame_overlap --shift-to-zero=false \
      ${data_dir}/segments ark:- \| \
    segmentation-combine-segments-to-recordings ark:- \
      "ark,t:utils/data/get_reco2utt_for_data.sh ${data_dir} |" ark:- \| \
    segmentation-create-subsegments --filter-label=1 --subsegment-label=0 \
      "ark:segmentation-init-from-lengths --label=1 ark,t:${whole_data_dir}/utt2num_frames ark:- |" \
      ark:- ark:- \| \
    segmentation-post-process --remove-labels=0 --max-segment-length=1000 \
      --post-process-label=1 --overlap-length=50 ark:- ark:- \| \
    segmentation-to-segments --single-speaker=true \
      --frame-shift=$frame_shift --frame-overlap=$frame_overlap \
      ark:- ark,t:$outside_data_dir/utt2spk $outside_data_dir/segments

  utils/fix_data_dir.sh $outside_data_dir
  
  utils/data/subsegment_data_dir.sh $whole_data_dir $outside_data_dir/segments \
    $outside_data_dir/tmp
  cp $outside_data_dir/tmp/feats.scp $outside_data_dir
fi

extended_data_dir=$dir/${data_id}_extended
if [ $stage -le 6 ]; then
  cp $dir/${data_id}_manual_segments/cmvn.scp ${outside_data_dir} || exit 1
  utils/fix_data_dir.sh $outside_data_dir
  
  utils/combine_data.sh $extended_data_dir $data_dir $outside_data_dir
fi

###############################################################################
# Create graph for decoding
###############################################################################

# TODO: By default, we use word LM. If required, we can think 
# consider phone LM.
  
model_id=`basename $model_dir`

graph_dir=$dir/$model_id/graph
if [ $stage -le 7 ]; then
  if [ ! -f $graph_dir/HCLG.fst ]; then
    cp -rT $lang_test/ $dir/lang_test
    utils/mkgraph.sh $dir/lang_test $model_dir $graph_dir || exit 1
  fi
fi

###############################################################################
# Decode extended data directory
###############################################################################

decode_dir=$dir/${model_id}/decode_${data_id}_extended
# Decode without lattice (get only best path)
if [ $stage -le 8 ]; then
  mkdir -p $decode_dir
  cp $model_dir/{final.mdl,final.mat,*_opts,tree} $dir/${model_id}
  cp $model_dir/phones.txt $dir/$model_id

  steps/decode_nolats.sh --cmd "$cmd --mem 2G" --nj $nj \
    --max-active 1000 --beam 10.0 --write-words false \
    --write-alignments true \
    $graph_dir ${extended_data_dir} $decode_dir
fi

# Get VAD based on the decoded best path
decode_vad_dir=$dir/${model_id}_decode_vad_${data_id}
if [ $stage -le 9 ]; then
  cp $dir/$model_id/final.mdl $decode_dir
  steps/segmentation/internal/convert_ali_to_vad.sh --cmd "$cmd" \
    ${decode_dir} $dir/sad_map $decode_vad_dir
fi

[ ! -s $decode_vad_dir/sad_seg.scp ] && echo "$0: $decode_vad_dir/vad.scp is empty" && exit 1

vad_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $vad_dir ${PWD}`

# Assume that the regions outside manual segments is all silence.
if [ $stage -le 10 ]; then
  segmentation-init-from-segments --frame-shift=$frame_shift \
    --frame-overlap=$frame_overlap --label=0 \
    $outside_data_dir/segments \
    ark,scp:$vad_dir/outside_sad_seg.ark,$vad_dir/outside_sad_seg.scp
fi

reco_vad_dir=$dir/${model_id}_reco_vad_${data_id}
mkdir -p $reco_vad_dir
if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $reco_vad_dir/storage ]; then
  utils/create_split_dir.pl \
    /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$reco_vad_dir/storage $reco_vad_dir/storage
fi

reco_vad_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $reco_vad_dir ${PWD}`

echo $reco_nj > $reco_vad_dir/num_jobs

if [ $stage -le 11 ]; then
  utils/data/get_reco2utt_for_data.sh $extended_data_dir > $reco_vad_dir/reco2utt
  splits=
  for n in `seq $reco_nj`; do
    splits="$splits $reco_vad_dir/reco2utt.$n.$reco_nj"
  done
  utils/split_scp.pl $reco_vad_dir/reco2utt $splits
  
  for n in `seq $reco_nj`; do
    utils/spk2utt_to_utt2spk.pl $reco_vad_dir/reco2utt.$n.$reco_nj > $reco_vad_dir/utt2reco.$n.$reco_nj
  done

  $cmd JOB=1:$reco_nj $reco_vad_dir/log/intersect_vad.JOB.log \
    segmentation-intersect-segments --mismatch-label=1000 \
      "scp:cat $vad_dir/sad_seg.scp $vad_dir/outside_sad_seg.scp | sort -k1,1 | utils/filter_scp.pl $reco_vad_dir/utt2reco.JOB.$reco_nj |" \
      "scp:utils/filter_scp.pl $reco_vad_dir/utt2reco.JOB.$reco_nj $decode_vad_dir/sad_seg.scp |" \
      ark:- \| \
    segmentation-post-process --remove-labels=1000 \
      --merge-adjacent-segments --max-intersegment-length=10 ark:- ark:- \| \
    segmentation-combine-segments ark:- \
      "ark:utils/filter_scp.pl $reco_vad_dir/utt2reco.JOB.$reco_nj $extended_data_dir/segments | segmentation-init-from-segments --shift-to-zero=false - ark:- |" \
      ark,t:$reco_vad_dir/reco2utt.JOB.$reco_nj \
      ark,scp:$reco_vad_dir/sad_seg.JOB.ark,$reco_vad_dir/sad_seg.JOB.scp

  for n in `seq $reco_nj`; do
    cat $reco_vad_dir/sad_seg.$n.scp
  done > $reco_vad_dir/sad_seg.scp
fi

set +e 
for n in `seq $reco_nj`; do
  utils/create_data_link.pl $reco_vad_dir/deriv_weights.$n.ark
  utils/create_data_link.pl $reco_vad_dir/deriv_weights_for_uncorrupted.$n.ark
  utils/create_data_link.pl $reco_vad_dir/speech_labels.$n.ark
done
set -e

# Deriv weights to train only on "good" frames, i.e. where alignment and decoding match
if [ $stage -le 12 ]; then
  $cmd JOB=1:$reco_nj $reco_vad_dir/log/get_deriv_weights.JOB.log \
    segmentation-post-process --merge-labels=0:1:2:3 --merge-dst-label=1 \
      scp:$reco_vad_dir/sad_seg.JOB.scp ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:${whole_data_dir}/utt2num_frames ark:- ark,t:- \| \
    steps/segmentation/convert_ali_to_vec.pl \| copy-vector ark,t:- \
      ark,scp:$reco_vad_dir/deriv_weights.JOB.ark,$reco_vad_dir/deriv_weights.JOB.scp
  
  for n in `seq $reco_nj`; do
    cat $reco_vad_dir/deriv_weights.$n.scp
  done > $reco_vad_dir/deriv_weights.scp
fi

# Deriv weights to train only on silence frames
if [ $stage -le 13 ]; then
  $cmd JOB=1:$reco_nj $reco_vad_dir/log/get_deriv_weights_for_uncorrupted.JOB.log \
    segmentation-post-process --remove-labels=1:2:3 scp:$reco_vad_dir/sad_seg.JOB.scp ark:- \| \
    segmentation-post-process --merge-labels=0 --merge-dst-label=1 ark:- ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:${whole_data_dir}/utt2num_frames ark:- ark,t:- \| \
    steps/segmentation/convert_ali_to_vec.pl \| copy-vector ark,t:- \
      ark,scp:$reco_vad_dir/deriv_weights_for_uncorrupted.JOB.ark,$reco_vad_dir/deriv_weights_for_uncorrupted.JOB.scp

  for n in `seq $reco_nj`; do
    cat $reco_vad_dir/deriv_weights_for_uncorrupted.$n.scp
  done > $reco_vad_dir/deriv_weights_for_uncorrupted.scp
fi

# Get per-frame SAD labels at recording-level
if [ $stage -le 14 ]; then
  $cmd JOB=1:$reco_nj $reco_vad_dir/log/get_speech_labels.JOB.log \
    segmentation-copy --keep-label=1 scp:$reco_vad_dir/sad_seg.JOB.scp ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:${whole_data_dir}/utt2num_frames \
      ark:- ark,scp:$reco_vad_dir/speech_labels.JOB.ark,$reco_vad_dir/speech_labels.JOB.scp

  for n in `seq $reco_nj`; do
    cat $reco_vad_dir/speech_labels.$n.scp
  done > $reco_vad_dir/speech_labels.scp
fi

# Deriv weights to train only on manual segments
if [ $stage -le 15 ]; then
  $cmd JOB=1:$reco_nj $reco_vad_dir/log/convert_manual_segments_to_deriv_weights.JOB.log \
    segmentation-init-from-segments --shift-to-zero=false --frame-shift=$frame_shift --frame-overlap=$frame_overlap \
      "utils/filter_scp.pl $reco_vad_dir/utt2reco.JOB.$reco_nj $data_dir/segments |" ark:- \| \
    segmentation-combine-segments-to-recordings ark:- \
      "ark,t:utils/data/get_reco2utt_for_data.sh $data_dir | utils/filter_scp.pl $reco_vad_dir/reco2utt.JOB.$reco_nj |" \
      ark:- \| \
    segmentation-to-ali --lengths-rspecifier=ark,t:${whole_data_dir}/utt2num_frames \
      ark:- ark,t:- \| \
    steps/segmentation/convert_ali_to_vec.pl \| copy-vector ark,t:- \
      ark,scp:$reco_vad_dir/deriv_weights_manual_seg.JOB.ark,$reco_vad_dir/deriv_weights_manual_seg.JOB.scp

  for n in `seq $reco_nj`; do
    cat $reco_vad_dir/deriv_weights_manual_seg.$n.scp
  done > $reco_vad_dir/deriv_weights_manual_seg.scp
fi

echo "$0: Finished creating corpus for training Universal SAD with data in $whole_data_dir and labels in $reco_vad_dir"
