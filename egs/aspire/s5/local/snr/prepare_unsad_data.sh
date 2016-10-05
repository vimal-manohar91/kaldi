#!/bin/bash

set -u
set -o pipefail

. path.sh

stage=-2
cmd=queue.pl
reco_nj=40
nj=100

#map_noise_to_sil=true
#map_unknown_to_speech=true
phone_map=

feat_type=mfcc        # mfcc or plp
add_pitch=false       # Add pitch features

config_dir=conf
feat_config=
pitch_config=     

outside_keep_proportion=1.0
get_whole_recordings_and_weights=true

mfccdir=mfcc
plpdir=plp

speed_perturb=true
sat_model_dir=

. utils/parse_options.sh

if [ $# -ne 5 ]; then
  echo "This script takes a data directory and creates a new data directory "
  echo "and speech activity labels "
  echo "for the purpose of training a Universal Speech Activity Detector."
  echo "Usage: $0 [options] <data-dir> <lang> <ali-dir> <model-dir> <temp-dir>"
  echo " e.g.: $0 data/train_100k data/lang exp/tri4a_ali_100k exp/tri4a exp/vad_data_prep"
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
ali_dir=$3
model_dir=$4
dir=$5

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
    steps/make_mfcc_pitch.sh --cmd "$cmd" --nj $nj \
      --mfcc-config $mfcc_config --pitch-config $pitch_config $1 $2 $3 || exit 1
  else
    steps/make_mfcc.sh --cmd "$cmd" --nj $nj \
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
    steps/make_plp_pitch.sh --cmd "$cmd" --nj $nj \
      --plp-config $plp_config --pitch-config $pitch_config $1 $2 $3 || exit 1
  else
    steps/make_plp.sh --cmd "$cmd" --nj $nj \
      --plp-config $plp_config $1 $2 $3 || exit 1
  fi
}

steps/segmentation/get_sad_map.py \
  --init-phone-map="$phone_map" \
  --map-noise-to-sil=$map_noise_to_sil \
  --map-unk-to-speech=$map_unk_to_speech \
  --unk=$oov \
  $lang | utils/sym2int.pl -f 1 $lang/phones.txt > $dir/phone_map

whole_data_dir=${data_dir}_whole
whole_data_id=${data_id}_whole

utils/convert_data_dir_to_whole.sh ${data_dir} ${whole_data_dir}
utils/get_utt2dur.sh ${whole_data_dir}

data_id=$(basename $data_dir)

if $speed_perturb; then
  plpdir=${plpdir}_sp
  mfccdir=${mfccdir}_sp

  utils/data/perturb_data_dir_speed_3way.sh ${whole_data_dir} ${whole_data_dir}_sp
  utils/data/perturb_data_dir_speed_3way.sh ${data_dir} ${data_dir}_sp
 
  if [ $stage -le 1 ]; then
    #utils/perturb_data_dir_speed.sh 0.9 ${data_dir} ${data_dir}_temp1
    #utils/perturb_data_dir_speed.sh 1.1 ${data_dir} ${data_dir}_temp2
    #utils/perturb_data_dir_speed.sh 1.0 ${data_dir} ${data_dir}_temp0
    #utils/combine_data.sh ${data_dir}_sp ${data_dir}_temp1 ${data_dir}_temp2 ${data_dir}_temp0
    #utils/validate_data_dir.sh --no-feats ${data_dir}_sp
    #rm -r ${data_dir}_temp1 ${data_dir}_temp2 ${data_dir}_temp0

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

  utils/data/subsegment_feats.sh ${whole_data_dir}_sp/feats.scp $frame_shift ${data_dir}_sp/segments > ${data_dir}_sp/feats.scp

  if [ -z "$sat_model_dir" ]; then
    ali_dir=${model_dir}_ali_sp
    if [ $stage -le 1 ]; then
      steps/align_si.sh --nj $nj --cmd "$cmd" \
        ${data_dir}_sp ${lang} ${model_dir} ${model_dir}_ali_sp || exit 1
    fi
  else
    ali_dir=${sat_model_dir}_ali_sp
    #obtain the alignment of the perturbed data
    if [ $stage -le 1 ]; then
      steps/align_fmllr.sh --nj $nj --cmd "$cmd" \
        ${data_dir}_sp ${lang} ${sat_model_dir} ${sat_model_dir}_ali_sp || exit 1
    fi
  fi
    
  data_dir=${data_dir}_sp
  whole_data_dir=${whole_data_dir}_sp
fi

# All the data from this point is speed perturbed.

data_id=$(basename $data_dir)
utils/split_data.sh $data_dir $nj

###############################################################################
# Convert alignment for the provided segments into 
# initial SAD labels at utterance-level in segmentation format
###############################################################################

vad_dir=$dir/`basename ${ali_dir}`_vad_${data_id}
if [ $stage -le 2 ]; then
  steps/segmentation/internal/convert_ali_to_vad.sh --cmd "$cmd" \
    $data_dir $ali_dir \
    $phone_map $vad_dir
fi

[ ! -s $vad_dir/sad_seg.scp ] && echo "$0: $vad_dir/vad.scp is empty" && exit 1
  
#utils/split_data.sh --per-reco $data_dir $reco_nj
#segmentation-combine-segments ark,s:$vad_dir/sad_seg.scp 
#  "ark,s:segmentation-init-from-segments --shift-to-zero=false --frame-shift=$ali_frame_shift --frame-overlap=$ali_frame_overlap ${data}/split${reco_nj}reco/JOB/segments ark:- |" \
#  "ark:cat ${data}/split${reco_nj}reco/JOB/segments | cut -d ' ' -f 1,2 | utils/utt2spk_to_spk2utt.pl | sort -k1,1 |" ark:- 

###############################################################################

###############################################################################
# Compute length of recording
###############################################################################

frame_shift_info=`cat $feat_config | steps/segmentation/get_frame_shift_from_config.pl`

frame_shift=`echo $frame_shift_info | awk '{print $1}'`
frame_overlap=`echo $frame_shift_info | awk '{print $2}'`

awk -v fs=$frame_shift fovlp=$frame_overlap \
  '{print $1" "int( ($2 - fovlp) / fs)}' $whole_data_dir/utt2dur \
  > $whole_data_dir/utt2num_frames


# Create extended data directory that consists of the provided 
# segments along with the segments outside it.
# This is basically dividing the whole recording into pieces
# consisting of pieces corresponding to the provided segments 
# and outside the provided segments.

###############################################################################
# Create segments outside of the manual segments
###############################################################################

outside_data_dir=$dir/${data_id}_outside
if [ $stage -le 3 ]; then
  rm -rf $outside_data_dir
  mkdir -p $outside_data_dir/split${reco_nj}reco

  for f in wav.scp reco2file_and_channel stm glm; do 
    [ -f ${data_dir}/$f ] && cp ${data_dir}/outside_data_dir
  done

  utils/copy_data_dir.sh $data_dir $outside_data_dir
  for f in cmvn.scp feats.scp text utt2uniq utt2dur; do
    rm -f $outside_data_dir/$f
  done

  utils/split_data.sh --per-reco $data_dir $reco_nj
  utils/split_data.sh --per-reco ${whole_data_dir} $reco_nj

  for n in `seq $reco_nj`; do
    awk '{print $2}' ${whole_data_dir}/split${reco_nj}reco/$n/segments | \
      utils/filter_scp.pl /dev/stdin $whole_data_dir/utt2num_frames > \
      ${whole_data_dir}/split${reco_nj}reco/$n/utt2num_frames
  done

  utils/get_reco2utt.sh $data_dir

  $cmd JOB=1:$reco_nj $dir/log/get_empty_segments.JOB.log \
    segmentation-init-from-segments --frame-shift=$frame_shift \
    --frame-overlap=$frame_overlap --shift-to-zero=false \
    ${data_dir}/split${reco_nj}reco/JOB/segments ark:- \| \
    segmentation-combine-segments-to-recording ark:- \
    "ark,t:cut -d ' ' -f 1,2 ${data_dir}/split${reco_nj}/reco/JOB/segments  | utils/utt2spk_to_spk2utt.pl |" ark:- \| \
    segmentation-create-subsegments --secondary-label=1 --subsegment-label=0 \
    "ark:segmentation-init-from-lengths --label=1 ark,t:${whole_data_dir}/split${reco_nj}reco/JOB/utt2num_frames ark:- |" \
    ark:- ark:- \| \
    segmentation-post-process --remove-labels=0 --max-segment-length=1000 \
    --post-process-label=1 --overlap-length=50 \
    ark:- ark:- \| segmentation-to-segments --single-speaker=true \
    --frame-shift=$frame_shift --frame-overlap=$frame_overlap \
    ark:- ark,t:$outside_data_dir/split${reco_nj}reco/JOB/utt2spk \
    $outside_data_dir/split${reco_nj}reco/JOB/segments || exit 1

  for n in `seq $reco_nj`; do
    cat $outside_data_dir/split${reco_nj}reco/$n/utt2spk
  done | sort -k1,1 > $outside_data_dir/utt2spk
  
  for n in `seq $reco_nj`; do
    cat $outside_data_dir/split${reco_nj}reco/$n/segments
  done | sort -k1,1 > $outside_data_dir/segments

  utils/fix_data_dir.sh $outside_data_dir
  
  utils/data/subsegment_feats.sh ${whole_data_dir}_sp/feats.scp $frame_shift ${outside_data_dir}_sp/segments > ${outside_data_dir}_sp/feats.scp
fi

###############################################################################
# Create graph for decoding
###############################################################################

# TODO: By default, we use word LM. If required, we can think 
# consider phone LM.
graph_dir=$model_dir/graph
if [ $stage -le 4 ]; then
  if [ ! -d $graph_dir ]; then
    utils/mkgraph.sh ${lang} $model_dir $graph_dir || exit 1
  fi
fi

###############################################################################
# Decode extended data directory
###############################################################################

extended_data_dir=$dir/${data_id}_extended
steps/combine_data.sh $extended_data_dir $data_dir $outside_data_dir

utils/split_data.sh --per-reco $extended_data_dir $reco_nj
for n in `seq $reco_nj`; do
  awk '{print $1" "$2}' ${extended_data_dir}/split${reco_nj}reco/$n/segments | \
    utils/utt2spk_to_spk2utt.pl \
    ${extended_data_dir}/split${reco_nj}reco/$n/reco2utt
done

# Decode without lattice (get only best path)
if [ $stage -le 5 ]; then
  steps/decode_nolats.sh --cmd "$cmd --mem 2G" --nj $nj \
    --max-active 1000 --beam 10.0 --write-words false \
    --write-alignments true \
    $graph_dir ${extended_data_dir} \
    ${model_dir}/decode_${data_id}_extended || exit 1
fi

# Get VAD based on the decoded best path
decode_vad_dir=$dir/${model_dir}_decode_vad_${data_id}
if [ $stage -le 6 ]; then
  steps/segmentation/internal/convert_ali_to_vad.sh --cmd "$cmd" \
    $extended_data_dir ${model_dir}/decode_${data_id}_extended \
    $phone_map $decode_vad_dir
fi

[ ! -s $decode_vad_dir/sad_seg.scp ] && echo "$0: $decode_vad_dir/vad.scp is empty" && exit 1

if [ $stage -le 7 ]; then
  segmentation-init-from-segments --frame-shift=$frame_shift \
    --frame-overlap=$frame_overlap --segment-label=0 \
    $outside_data_dir/segments \
    ark,scp:$vad_dir/outside_sad_seg.ark,$vad_dir/outside_sad_seg.scp
fi

reco_vad_dir=$dir/${model_dir}_reco_vad_${data_id}
if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
  utils/create_split_dir.pl \
    /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$reco_vad_dir/storage $reco_vad_dir/storage
fi

reco_vad_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $reco_vad_dir ${PWD}`

echo $reco_nj > $reco_vad_dir/num_jobs

if [ $stage -le 8 ]; then
  $cmd JOB=1:$reco_nj $reco_vad_dir/log/intersect_vad.JOB.log \
    segmentation-intersect-segments --mismatch-label=10 \
    "scp:cat $vad_dir/sad_seg.scp $vad_dir/outside_sad_seg | sort -k1,1 | utils/filter_scp.pl $extended_data_dir/split${reco_nj}reco/JOB/utt2spk |" \
    "scp:utils/filter_scp.pl $extended_data_dir/split${reco_nj}reco/JOB/utt2spk $decode_dir/sad_seg.scp |" \
    ark:- \| segmentation-post-process --remove-labels=10 \
    --merge-adjacent-segments --max-intersegment-length=10 ark:- ark:- \| \
    segmentation-combine-segments ark:- "ark:segmentation-init-from-segments --shift-to-zero=false $extended_data_dir/split${reco_nj}reco/JOB/segments |" ark,t:$extended_data_dir/split${reco_nj}reco/utt2reco ark:- \
    ark,scp:$reco_vad_dir/sad_seg.JOB.ark,$reco_vad_dir/sad_seg.JOB.scp
  for n in `seq $reco_nj`; do
    cat $reco_vad_dir/sad_seg.$n.scp
  done > $reco_vad_dir/sad_seg.scp
fi

if [ $stage -le 9 ]; then
  for n in `seq $reco_nj`; do
    utils/create_data_link.pl $reco_vad_dir/deriv_weights.$n.ark
    utils/create_data_link.pl $reco_vad_dir/deriv_weights_for_uncorrupted.$n.ark
    utils/create_data_link.pl $reco_vad_dir/speech_feat.$n.ark
  done

  $cmd JOB=1:$reco_nj $reco_vad_dir/log/get_deriv_weights.JOB.log \
    segmentation-post-process --merge-labels=0:1:2:3 --merge-dst-label=1 \
    scp:$reco_vad_dir/sad_seg.JOB.scp ark:- \| \
    segmentation-to-ali --lengths-rspecifier=${whole_data_dir}/utt2num_frames ark:- ark,t:- \| \
    perl -pe 's/\[|\]//g' \| vector-to-feat ark:- ark:- \| copy-feats --compress \
    ark:- ark,scp:$reco_vad_dir/deriv_weights.JOB.ark,$reco_vad_dir/deriv_weights.JOB.scp
  
  for n in `seq $reco_nj`; do
    cat $reco_vad_dir/deriv_weights.$n.scp
  done > $reco_vad_dir/deriv_weights.scp
  
  $cmd JOB=1:$reco_nj $reco_vad_dir/log/get_deriv_weights_for_uncorrupted.JOB.log \
    segmentation-post-process --remove-labels=1:2:3 scp:$reco_vad_dir/sad_seg.JOB.scp \
    ark:- \| segmentation-post-process --merge-labels=0 --merge-dst-label=1 ark:- ark:- \| \
    segmentation-to-ali --lengths-rspecifier=${whole_data_dir}/utt2num_frames ark:- ark,t:- \| \
    perl -pe 's/\[|\]//g' \| vector-to-feat ark:- ark:- \| copy-feats --compress \
    ark:- ark,scp:$reco_vad_dir/deriv_weights_for_uncorrupted.JOB.ark,$reco_vad_dir/deriv_weights_for_uncorrupted.JOB.scp
  for n in `seq $reco_nj`; do
    cat $reco_vad_dir/deriv_weights_for_uncorrupted.$n.scp
  done > $reco_vad_dir/deriv_weights_for_uncorrupted.scp

  $cmd JOB=1:$reco_nj $reco_vad_dir/log/get_speech_labels.JOB.log \
    segmentation-to-ali --lengths-rspecifier=${whole_data_dir}/utt2num_frames \
    scp:$reco_vad_dir/sad_seg.JOB.scp ark,t:- \| \
    perl -pe 's/\[|\]//g' \| vector-to-feat ark:- ark:- \| copy-feats --compress \
    ark:- ark,scp:$reco_vad_dir/speech_feat.JOB.ark,$reco_vad_dir/speech_feat.JOB.scp
fi

echo "$0: Finished creating corpus for training Universal SAD with data in $whole_data_dir and labels in $reco_vad_dir"
