#!/bin/bash

set -u
set -o pipefail
set -e

. path.sh

stage=-2
reco_nj=40
nj=100
cmd=queue.pl
map_noise_to_sil=true
map_unknown_to_speech=true
feat_type=mfcc
add_pitch=false
pitch_config=
phone_map=
feat_config=
config_dir=conf
mfccdir=mfcc
plpdir=plp
speed_perturb=false
. utils/parse_options.sh

if [ $# -ne 5 ]; then
  echo "This script takes a data directory and creates a new data directory "
  echo "and speech activity labels "
  echo "for the purpose of training a Universal Speech Activity Detector."
  echo "Usage: $0 [options] <data-dir> <lang> <model-dir> <out-data-dir> <dir>"
  echo " e.g.: $0 data/train_100k data/lang exp/tri4a_ali_100k exp/vad_data_prep"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --file-nj <#njobs|4>                             # Split a whole data directory into these many pieces"
  echo "  --nj      <#njobs|4>                             # Split a segmented data directory into these many pieces"
  exit 1
fi

data_dir=$1
lang=$2
model_dir=$3
out_data_dir=$4
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

if [ -z "$phone_map" ]; then
  phone_map=$dir/phone_map
  oov=`cat $lang/oov.txt 2>/dev/null` || exit 1
  {
    cat $lang/phones/nonsilence.txt | awk '{print $1" "1}';
    cat $lang/phones/silence.txt | awk -v oov=$oov '{if ($1 != oov) print $1" "0; else print $1" "3; }';
  } | awk -v map_noise_to_sil=$map_noise_to_sil -v map_unknown_to_speech=$map_unknown_to_speech \
    '{if ($2 == 2 && map_noise_to_sil == "true") print $1" 0"; 
      else if ($2 == 3 && map_unknown_to_speech) print $1" 1";
      else print $0;}' > $dir/phone_map
elif $map_noise_to_sil || $map_unknown_to_speech; then
  cat $phone_map | \
    awk -v map_noise_to_sil=$map_noise_to_sil -v map_unknown_to_speech=$map_unknown_to_speech \
    '{if ($2 == 2 && map_noise_to_sil == "true") print $1" 0"; 
      else if ($2 == 3 && map_unknown_to_speech) print $1" 1";
      else print $0;}' > \
    $dir/phone_map
  phone_map=$dir/phone_map
fi

if [ $stage -le 0 ]; then
  diarization/convert_data_dir_to_whole.sh $data_dir ${data_dir}_whole
fi

data_dir=${data_dir}_whole
data_id=$(basename $data_dir)

if $speed_perturb; then
  plpdir=${plpdir}_sp
  mfccdir=${mfccdir}_sp
 
  if [ $stage -le 1 ]; then
    utils/perturb_data_dir_speed.sh 0.9 ${data_dir} ${data_dir}_temp1
    utils/perturb_data_dir_speed.sh 1.1 ${data_dir} ${data_dir}_temp2
    utils/perturb_data_dir_speed.sh 1.0 ${data_dir} ${data_dir}_temp0
    utils/combine_data.sh ${data_dir}_sp ${data_dir}_temp1 ${data_dir}_temp2 ${data_dir}_temp0
    utils/validate_data_dir.sh --no-feats --no-text ${data_dir}_sp
    rm -r ${data_dir}_temp1 ${data_dir}_temp2 ${data_dir}_temp0

    if [ $feat_type == "mfcc" ]; then
      if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
        utils/create_split_dir.pl \
          /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
      fi
      make_mfcc --cmd "$cmd --max-jobs-run 40" --nj $nj \
        --mfcc-config $feat_config \
        --add-pitch $add_pitch --pitch-config $pitch_config \
        ${data_dir}_sp exp/make_mfcc/${data_id}_sp $mfccdir || exit 1
      steps/compute_cmvn_stats.sh \
        ${data_dir}_sp exp/make_mfcc/${data_id}_sp $mfccdir || exit 1
    elif [ $feat_type == "plp" ]; then
      if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $plpdir/storage ]; then
        utils/create_split_dir.pl \
          /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$plpdir/storage $plpdir/storage
      fi

      make_plp --cmd "$cmd --max-jobs-run 40" --nj $nj \
        --plp-config $feat_config \
        --add-pitch $add_pitch --pitch-config $pitch_config \
        ${data_dir}_sp exp/make_plp/${data_id}_sp $plpdir || exit 1
      steps/compute_cmvn_stats.sh \
        ${data_dir}_sp exp/make_plp/${data_id}_sp $plpdir || exit 1
    fi
        
    utils/fix_data_dir.sh ${data_dir}_sp
  fi

  data_dir=${data_dir}_sp
  utils/copy_data_dir.sh --validate-opts "--no-text" ${data_dir} ${out_data_dir}
fi

# By default, we use word LM. If required, we can think 
# consider phone LM
graph_dir=$model_dir/graph
if [ $stage -le 2 ]; then
  if [ ! -d $graph_dir ]; then
    utils/mkgraph.sh ${lang} $model_dir $graph_dir || exit 1
  fi
fi

# Decode without lattice (get only best path)
if [ $stage -le 3 ]; then
  steps/decode_nolats.sh --cmd "$cmd --mem 2G" --nj $nj \
    --max-active 1000 --beam 10.0 --write-words false \
    --write-alignments true \
    $graph_dir ${data_dir} \
    ${model_dir}/decode_${data_id} || exit 1
fi

# Get VAD based on the decoded best path
decode_vad_dir=$dir/${model_dir}_decode_vad_${data_id}
if [ $stage -le 4 ]; then
  diarization/convert_ali_to_vad.sh --phone-map $phone_map \
    --cmd "$cmd" --model $model_dir/final.mdl \
    $data_dir $graph_dir \
  $model_dir/decode_${data_id} $decode_vad_dir || exit 1
fi

if [ $stage -le 10 ]; then
  mkdir -p $dir/reco_vad
  cp $decode_vad_dir/vad.scp $dir/reco_vad/vad.scp
fi

echo "$0: Finished creating corpus for training Universal SAD"

