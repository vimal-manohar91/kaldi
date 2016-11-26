#!/bin/bash
set -e
set -u
set -o pipefail

. path.sh
. cmd.sh

num_data_reps=5
data_dir=data/train_si284
dest_wav_dir=wavs

nj=40
reco_nj=40

stage=0
corruption_stage=-10

pad_silence=false

mfcc_config=conf/mfcc_hires_bp_vh.conf
feat_suffix=hires_bp_vh
mfcc_irm_config=conf/mfcc_hires_bp.conf

data_only=false
corrupt_only=false
speed_perturb=true

reco_vad_dir=

max_jobs_run=20

snrs="20:10:15:5:0:-5"
foreground_snrs="20:10:15:5:0:-5"
background_snrs="20:10:15:5:0:-5"
base_rirs=simulated

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  exit 1
fi

data_id=`basename ${data_dir}`

rvb_opts=()
if [ "$base_rirs" == "simulated" ]; then
  # This is the config for the system using simulated RIRs and point-source noises
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
  rvb_opts+=(--noise-set-parameters RIRS_NOISES/pointsource_noises/noise_list)
else
  # This is the config for the JHU ASpIRE submission system
  rvb_opts+=(--rir-set-parameters "1.0, RIRS_NOISES/real_rirs_isotropic_noises/rir_list")
  rvb_opts+=(--noise-set-parameters RIRS_NOISES/real_rirs_isotropic_noises/noise_list)
fi

corrupted_data_id=${data_id}_corrupted
clean_data_id=${data_id}_clean
noise_data_id=${data_id}_noise

if [ $stage -le 1 ]; then
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --prefix="rev" \
    --foreground-snrs=$foreground_snrs \
    --background-snrs=$background_snrs \
    --speech-rvb-probability=1 \
    --pointsource-noise-addition-probability=1 \
    --isotropic-noise-addition-probability=1 \
    --num-replications=$num_data_reps \
    --max-noises-per-minute=1 \
    --output-additive-noise-dir=data/${noise_data_id} \
    --output-reverb-dir=data/${clean_data_id} \
    data/${data_id} data/${corrupted_data_id}
fi

clean_data_dir=data/${clean_data_id}
corrupted_data_dir=data/${corrupted_data_id}
noise_data_dir=data/${noise_data_id}

if $speed_perturb; then
  if [ $stage -le 2 ]; then
    ## Assuming whole data directories
    for x in $clean_data_dir $corrupted_data_dir $noise_data_dir; do
      cp $x/reco2dur $x/utt2dur
      utils/data/perturb_data_dir_speed_3way.sh $x ${x}_sp
    done
  fi

  corrupted_data_dir=${corrupted_data_dir}_sp
  clean_data_dir=${clean_data_dir}_sp
  noise_data_dir=${noise_data_dir}_sp

  corrupted_data_id=${corrupted_data_id}_sp
  clean_data_id=${clean_data_id}_sp
  noise_data_id=${noise_data_id}_sp

  if [ $stage -le 3 ]; then
    utils/data/perturb_data_dir_volume.sh --scale-low 0.03125 --scale-high 2 ${corrupted_data_dir}
    utils/data/perturb_data_dir_volume.sh --reco2vol ${corrupted_data_dir}/reco2vol ${clean_data_dir}
    utils/data/perturb_data_dir_volume.sh --reco2vol ${corrupted_data_dir}/reco2vol ${noise_data_dir}
  fi
fi

if $corrupt_only; then
  echo "$0: Got corrupted data directory in ${corrupted_data_dir}"
  exit 0
fi

mfccdir=`basename $mfcc_config`
mfccdir=${mfccdir%%.conf}

if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
  utils/create_split_dir.pl \
    /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
fi

if [ $stage -le 4 ]; then
  if [ ! -z $feat_suffix ]; then
    utils/copy_data_dir.sh $corrupted_data_dir ${corrupted_data_dir}_$feat_suffix
    corrupted_data_dir=${corrupted_data_dir}_$feat_suffix
  fi
  steps/make_mfcc.sh --mfcc-config $mfcc_config \
    --cmd "$train_cmd" --nj $reco_nj \
    $corrupted_data_dir exp/make_${mfccdir}/${corrupted_data_id} $mfccdir
  steps/compute_cmvn_stats.sh --fake \
    $corrupted_data_dir exp/make_${mfccdir}/${corrupted_data_id} $mfccdir
else
  if [ ! -z $feat_suffix ]; then
    corrupted_data_dir=${corrupted_data_dir}_$feat_suffix
  fi
fi 

mfccdir=`basename $mfcc_irm_config`
mfccdir=${mfccdir%%.conf}

if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
  utils/create_split_dir.pl \
    /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
fi

if [ $stage -le 5 ]; then
  steps/make_mfcc.sh --mfcc-config $mfcc_irm_config \
    --cmd "$train_cmd" --nj $reco_nj \
    $clean_data_dir exp/make_${mfccdir}/${clean_data_id} $mfccdir
fi

if [ $stage -le 6 ]; then
  steps/make_mfcc.sh --mfcc-config $mfcc_irm_config \
    --cmd "$train_cmd" --nj $reco_nj \
    $noise_data_dir exp/make_${mfccdir}/${noise_data_id} $mfccdir
fi

targets_dir=irm_targets
if [ $stage -le 8 ]; then
  idct_opts=`cat $mfcc_irm_config | perl -e '
  my $num_mel_bins = 23;
  my $num_ceps = 13;
  my $cepstral_lifter = 22;
  while (<>) {
    if (m/--num-mel-bins=(\d+)/) {
      $num_mel_bins = $1;
    } 
    elsif (m/--num-ceps=(\d+)/) {
      $num_ceps = $1;
    } 
    elsif (m/--cepstral-lifter=(\d+)/) {
      $cepstral_lifter = $1;
    }
  }
  print ("--num-filters=$num_mel_bins --num-ceps=$num_ceps --cepstral-lifter=$cepstral_lifter\n");'`

  mkdir -p exp/make_irm_targets/${corrupted_data_id}

  utils/data/get_dct_matrix.py $idct_opts --get-idct-matrix=true exp/make_irm_targets/${corrupted_data_id}/idct.mat
  
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $targets_dir/storage ]; then
    utils/create_split_dir.pl \
      /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$targets_dir/storage $targets_dir/storage
  fi

  #ali_rspecifier=
  if [ ! -z "$reco_vad_dir" ]; then
    if [ ! -f $reco_vad_dir/speech_feat.scp ]; then
      echo "$0: Could not find file $reco_vad_dir/speech_feat.scp"
      exit 1
    fi
    
    cat $reco_vad_dir/speech_feat.scp | \
      steps/segmentation/get_reverb_scp.pl -f 1 $num_data_reps | \
      sort -k1,1 > ${corrupted_data_dir}/speech_feat.scp

    cat $reco_vad_dir/deriv_weights_manual_seg.scp | \
      steps/segmentation/get_reverb_scp.pl -f 1 $num_data_reps | \
      sort -k1,1 > ${corrupted_data_dir}/deriv_weights_manual_seg.scp

    #ali_rspecifier="ark,s,cs,t:utils/filter_scp.pl ${clean_data_dir}/split${nj}/JOB/utt2spk ${corrupted_data_dir}/speech_feat.scp | extract-column --column-index=0 scp:- ark,t:- | steps/segmentation/quantize_vector.pl |"
  fi
  
    #--silence-phones-str 0:2 --ali-rspecifier "$ali_rspecifier" \
  steps/segmentation/make_snr_targets.sh \
    --nj $nj --cmd "$train_cmd --max-jobs-run $max_jobs_run" \
    --target-type Irm --compress true --apply-exp false \
    --transform-matrix exp/make_irm_targets/${corrupted_data_id}/idct.mat \
    ${clean_data_dir} ${noise_data_dir} ${corrupted_data_dir} \
    exp/make_irm_targets/${corrupted_data_id} $targets_dir
fi

if [ $stage -le 9 ]; then
  cat $reco_vad_dir/deriv_weights.scp | \
    steps/segmentation/get_reverb_scp.pl -f 1 $num_data_reps | \
    sort -k1,1 > ${corrupted_data_dir}/deriv_weights.scp
fi

exit 0
