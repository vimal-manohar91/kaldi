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
mfcc_irm_config=conf/mfcc_hires_bp.conf

data_only=true
corrupt_only=true
dry_run=true
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
    utils/data/perturb_data_dir_volume.sh ${corrupted_data_dir}
    utils/data/perturb_data_dir_volume.sh --reco2vol ${corrupted_data_dir}/reco2vol ${clean_data_dir}
    utils/data/perturb_data_dir_volume.sh --reco2vol ${corrupted_data_dir}/reco2vol ${noise_data_dir}
  fi
fi

mfccdir=`basename $mfcc_config`
mfccdir=${mfccdir%%.conf}

if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
  utils/create_split_dir.pl \
    /export/b0{3,4,5,6}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
fi

if [ $stage -le 4 ]; then
  steps/make_mfcc.sh --mfcc-config $mfcc_config \
    --cmd "$train_cmd" --nj $reco_nj \
    $corrupted_data_dir exp/make_${mfccdir}/${corrupted_data_id} $mfccdir
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

  ali_rspecifier=
  if [ ! -z "$reco_vad_dir" ]; then
    if [ ! -f $reco_vad_dir/speech_feat.scp ]; then
      echo "$0: Could not find file $reco_vad_dir/speech_feat.scp"
      exit 1
    fi
    
    cat $reco_vad_dir/speech_feat.scp | \
      steps/segmentation/get_reverb_scp.pl -f 1 $num_data_reps | \
      sort -k1,1 > ${corrupted_data_dir}/speech_feat.scp

    ali_rspecifier="ark,s,cs,t:utils/filter_scp.pl ${clean_data_dir}/split${nj}/JOB/utt2spk ${corrupted_data_dir}/speech_feat.scp | extract-column --column-index=0 scp:- ark,t:- | steps/segmentation/quantize_vector.pl |"
  fi
  
  steps/segmentation/make_snr_targets.sh \
    --nj $nj --cmd "$train_cmd --max-jobs-run $max_jobs_run" \
    --target-type Irm --compress false --apply-exp true \
    --transform-matrix exp/make_irm_targets/${corrupted_data_id}/idct.mat \
    --silence-phones-str 0:2 --ali-rspecifier "$ali_rspecifier" \
    ${clean_data_dir} ${noise_data_dir} ${corrupted_data_dir} \
    exp/make_irm_targets/${corrupted_data_id} $targets_dir
fi


exit 0

####if [ $stage -le $num_data_reps ]; then
####  corrupted_data_dirs=
####  start_state=1
####  if [ $stage -gt 1 ]; then 
####    start_stage=$stage
####  fi
####  for x in `seq $start_stage $num_data_reps`; do
####    cur_dest_dir=data/temp_${data_id}_$x
####    output_clean_dir=data/temp_clean_${data_id}_$x
####    output_noise_dir=data/temp_noise_${data_id}_$x
####    local/snr/corrupt_data_dir.sh --dry-run $dry_run --random-seed $x --dest-wav-dir $dest_wav_dir/corrupted$x \
####      --output-clean-wav-dir $dest_wav_dir/clean$x --output-clean-dir $output_clean_dir \
####      --output-noise-wav-dir $dest_wav_dir/noise$x --output-noise-dir $output_noise_dir \
####      --pad-silence $pad_silence --stage $corruption_stage --tmp-dir exp/make_corrupt_$data_id/$x \
####      --nj $nj $data_dir data/impulse_noises $cur_dest_dir
####    corrupted_data_dirs+=" $cur_dest_dir"
####    clean_data_dirs+=" $output_clean_dir"
####    noise_data_dirs+=" $output_noise_dir"
####  done
####
####  rm -r ${data_dir}_{corrupted,clean,noise} || true
####  utils/combine_data.sh --extra-files utt2uniq ${data_dir}_corrupted ${corrupted_data_dirs}
####  utils/combine_data.sh --extra-files utt2uniq ${data_dir}_clean ${clean_data_dirs}
####  utils/combine_data.sh --extra-files utt2uniq ${data_dir}_noise ${noise_data_dirs}
####  
####  rm -r $corrupted_data_dirs || true
####  rm -r $clean_data_dirs || true
####fi
####
####if [ ! -z "$uncorrupted_vad_scp" ]; then
####  vad_scp=${data_dir}_corrupted/vad_multi.scp
####  for x in `seq 1 $num_data_reps`; do
####    awk -v x=$x '{print "corrupted"x"_"$0}' $uncorrupted_vad_scp > ${data_dir}_corrupted/vad_corrupted.scp
####    awk -v x=$x '{print "clean-"$0}' ${data_dir}_corrupted/vad_corrupted.scp 
####  done | cat - $uncorrupted_vad_scp | sort -k1,1 > ${data_dir}_corrupted/vad_multi.scp
####fi
####
####if $speed_perturb; then
####  if [ ! -z "$uncorrupted_vad_scp" ]; then
####    vad_scp=${data_dir}_corrupted/vad_multi_sp.scp
####    for x in 0.9 1.0 1.1; do
####      awk -v x=$x '{print "sp"x"-"$0}' $vad_scp > ${data_dir}_corrupted/vad_multi_sp.scp
####    done
####  fi
####
####  if [ $stage -le 11 ]; then
####    utils/data/perturb_data_dir_speed_3way.sh ${data_dir}_corrupted ${data_dir}_sp_corrupted
####    utils/data/perturb_data_dir_speed_3way.sh ${data_dir}_clean ${data_dir}_sp_clean
####    utils/data/perturb_data_dir_speed_3way.sh ${data_dir}_noise ${data_dir}_sp_noise
####
####    if $create_targets_for_uncorrupted_dir; then
####      utils/data/perturb_data_dir_speed_3way.sh ${data_dir} ${data_dir}_sp
####    fi
####  fi
####  data_dir=${data_dir}_sp
####fi
####
####data_id=`basename $data_dir`
####corrupted_data_dir=${data_dir}_corrupted
####corrupted_data_id=`basename $corrupted_data_dir`
####clean_data_dir=${data_dir}_clean
####clean_data_id=`basename $clean_data_dir`
####noise_data_dir=${data_dir}_noise
####noise_data_id=`basename $noise_data_dir`
####
####$corrupt_only && echo "--corrupt-only is true" && exit 1
####
####mfccdir=mfcc_hires
#####if [ $stage -le 2 ]; then
#####  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
#####    date=$(date +'%m_%d_%H_%M')
#####    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$mfccdir/storage $mfccdir/storage
#####  fi
#####
#####  utils/copy_data_dir.sh ${clean_data_dir} ${clean_data_dir}_hires
#####  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config $mfcc_config ${clean_data_dir}_hires exp/make_hires/${clean_data_id} mfcc_hires
#####  steps/compute_cmvn_stats.sh ${clean_data_dir}_hires exp/make_hires/${clean_data_id} mfcc_hires
#####  utils/fix_data_dir.sh ${clean_data_dir}_hires
#####fi
####
####fbankdir=fbank_feats
####if [ $stage -le 12 ]; then
####  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $fbankdir/storage ]; then
####    date=$(date +'%m_%d_%H_%M')
####    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$fbankdir/storage $fbankdir/storage
####  fi
####
####  rm -r ${clean_data_dir}_fbank || true
####  utils/copy_data_dir.sh ${clean_data_dir} ${clean_data_dir}_fbank
####  steps/make_fbank.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --fbank-config $fbank_config ${clean_data_dir}_fbank exp/make_fbank/${clean_data_id} fbank_feats 
####  steps/compute_cmvn_stats.sh --fake ${clean_data_dir}_fbank exp/make_fbank/${clean_data_id} fbank_feats
####  utils/fix_data_dir.sh ${clean_data_dir}_fbank
####fi
####
####if [ $stage -le 13 ]; then
####  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $fbankdir/storage ]; then
####    date=$(date +'%m_%d_%H_%M')
####    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$fbankdir/storage $fbankdir/storage
####  fi
####
####  rm -r ${noise_data_dir}_fbank || true
####  utils/copy_data_dir.sh ${noise_data_dir} ${noise_data_dir}_fbank
####  steps/make_fbank.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --fbank-config $fbank_config ${noise_data_dir}_fbank exp/make_fbank/${noise_data_id} fbank_feats 
####  steps/compute_cmvn_stats.sh --fake ${noise_data_dir}_fbank exp/make_fbank/${noise_data_id} fbank_feats
####  utils/fix_data_dir.sh ${noise_data_dir}_fbank
####fi
####
####if [ $stage -le 14 ]; then
####  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $fbankdir/storage ]; then
####    date=$(date +'%m_%d_%H_%M')
####    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$fbankdir/storage $fbankdir/storage
####  fi
####  
####  rm -r ${corrupted_data_dir}_fbank || true
####  utils/copy_data_dir.sh ${corrupted_data_dir} ${corrupted_data_dir}_fbank
####  steps/make_fbank.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --fbank-config $fbank_config ${corrupted_data_dir}_fbank exp/make_fbank/${corrupted_data_id} fbank_feats 
####  steps/compute_cmvn_stats.sh --fake ${corrupted_data_dir}_fbank exp/make_fbank/${corrupted_data_id} fbank_feats
####  utils/fix_data_dir.sh --utt-extra-files utt2uniq ${corrupted_data_dir}_fbank
####fi
####
####if $create_targets_for_uncorrupted_dir; then
####  if [ $stage -le 15 ]; then
####    rm -r ${data_dir}_fbank || true
####    utils/copy_data_dir.sh ${data_dir} ${data_dir}_fbank
####    steps/make_fbank.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --fbank-config $fbank_config ${data_dir}_fbank exp/make_fbank/${data_id} fbank_feats 
####    steps/compute_cmvn_stats.sh --fake ${data_dir}_fbank exp/make_fbank/${data_id} fbank_feats
####    utils/fix_data_dir.sh --utt-extra-files utt2uniq ${data_dir}_fbank
####  fi
####fi
####
####if [ $stage -le 16 ]; then
####  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
####    date=$(date +'%m_%d_%H_%M')
####    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$mfccdir/storage $mfccdir/storage
####  fi
####
####  rm -r ${corrupted_data_dir}_hires || true
####  utils/copy_data_dir.sh ${corrupted_data_dir} ${corrupted_data_dir}_hires
####  steps/make_mfcc.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --mfcc-config $mfcc_config ${corrupted_data_dir}_hires exp/make_hires/${corrupted_data_id} mfcc_hires
####  steps/compute_cmvn_stats.sh --fake ${corrupted_data_dir}_hires exp/make_hires/${corrupted_data_id} mfcc_hires
####  utils/fix_data_dir.sh --utt-extra-files utt2uniq ${corrupted_data_dir}_hires
####fi
####
####if $create_targets_for_uncorrupted_dir; then
####  if [ $stage -le 17 ]; then
####    rm -r ${clean_data_dir}_hires || true
####    utils/copy_data_dir.sh ${clean_data_dir} ${clean_data_dir}_hires
####    steps/make_mfcc.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --mfcc-config $mfcc_config ${clean_data_dir}_hires exp/make_hires/${clean_data_id} mfcc_hires 
####    steps/compute_cmvn_stats.sh --fake ${clean_data_dir}_hires exp/make_hires/${clean_data_id} mfcc_hires
####    utils/fix_data_dir.sh --utt-extra-files utt2uniq ${clean_data_dir}_hires
####  fi
####
####  if [ $stage -le 18 ]; then
####    utils/copy_data_dir.sh --utt-prefix "clean-" --spk-prefix "clean-" ${clean_data_dir}_fbank ${clean_data_dir}_clean_fbank
####    utils/copy_data_dir.sh --utt-prefix "clean-" --spk-prefix "clean-" ${clean_data_dir}_hires ${clean_data_dir}_clean_hires
####  fi
####  
####  if [ $stage -le 19 ]; then
####    rm -r ${data_dir}_hires || true
####    utils/copy_data_dir.sh ${data_dir} ${data_dir}_hires
####    steps/make_mfcc.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --mfcc-config $mfcc_config ${data_dir}_hires exp/make_hires/${data_id} mfcc_hires 
####    steps/compute_cmvn_stats.sh --fake ${data_dir}_hires exp/make_hires/${data_id} mfcc_hires
####    utils/fix_data_dir.sh --utt-extra-files utt2uniq ${data_dir}_hires
####  fi
####
####  if [ $stage -le 20 ]; then
####    utils/combine_data.sh --extra-files utt2uniq ${data_dir}_multi_fbank ${corrupted_data_dir} ${clean_data_dir}_clean_fbank ${data_dir}_fbank
####    utils/combine_data.sh --extra-files utt2uniq ${data_dir}_multi_hires ${corrupted_data_dir} ${clean_data_dir}_clean_hires ${data_dir}_hires
####  fi
####fi
####
####[ $(cat ${clean_data_dir}_fbank/utt2spk | wc -l) -ne $(cat ${corrupted_data_dir}_fbank/utt2spk | wc -l) ] && echo "$0: ${clean_data_dir}_fbank/utt2spk and ${corrupted_data_dir}_fbank/utt2spk have different number of lines" && exit 1
####
####[ $(cat ${noise_data_dir}_fbank/utt2spk | wc -l) -ne $(cat ${corrupted_data_dir}_fbank/utt2spk | wc -l) ] && echo "$0: ${noise_data_dir}_fbank/utt2spk and ${corrupted_data_dir}_fbank/utt2spk have different number of lines" && exit 1
####
####$data_only && echo "--data-only is true" && exit 1
####
####tmpdir=exp/make_irm_targets
####targets_dir=irm_targets
####  
####if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $targets_dir/storage ]; then
####  date=$(date +'%m_%d_%H_%M')
####  utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$targets_dir/storage $targets_dir/storage
####  for n in `seq $nj`; do 
####    utils/create_data_link.pl $targets_dir/${data_id}.$n.ark
####  done
####fi
####
####if $create_targets_for_uncorrupted_dir; then
####  if [ $stage -le 21 ]; then
####    local/snr/make_snr_targets.sh --length-tolerance 2 --compress false \
####      --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --target-type Irm --apply-exp true \
####      ${vad_scp:+--ali-rspecifier "scp:$vad_scp" --silence-phones-str "0:2:4:10"} \
####      --ignore-noise-dir true \
####      ${clean_data_dir}_fbank ${noise_data_dir}_fbank ${data_dir}_hires \
####      $tmpdir/$data_id $targets_dir || exit 1
####  fi
####
####  if [ $stage -le 22 ]; then
####    utils/copy_data_dir.sh --utt-prefix clean- --spk-prefix clean- --extra-files "irm_targets.scp" \
####      ${clean_data_dir}_hires ${clean_data_dir}_clean_hires || exit 1
####
####    local/snr/make_snr_targets.sh --length-tolerance 2 --compress false \
####      --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --target-type Irm --apply-exp true \
####      ${vad_scp:+--ali-rspecifier "scp:$vad_scp" --silence-phones-str "0:2:4:10"} \
####      --ignore-noise-dir true \
####      ${clean_data_dir}_clean_fbank ${noise_data_dir}_fbank ${clean_data_dir}_clean_hires \
####      $tmpdir/$data_id $targets_dir || exit 1
####  fi
####fi
####
####if [ $stage -le 23 ]; then
####  rm -r ${corrupted_data_dir}_hires || true
####  utils/copy_data_dir.sh ${corrupted_data_dir} ${corrupted_data_dir}_hires
####  steps/make_mfcc.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --mfcc-config $mfcc_config ${corrupted_data_dir}_hires exp/make_hires/${corrupted_data_id} mfcc_hires
####  steps/compute_cmvn_stats.sh --fake ${corrupted_data_dir}_hires exp/make_hires/${corrupted_data_id} mfcc_hires
####  utils/fix_data_dir.sh --utt-extra-files utt2uniq ${corrupted_data_dir}_hires
####fi
####
####if $create_targets_for_uncorrupted_dir; then
####  if [ $stage -le 23 ]; then
####  utils/combine_data.sh --extra-files utt2uniq ${data_dir}_multi_fbank ${corrupted_data_dir} ${clean_data_dir}_clean_fbank ${data_dir}_fbank
####  utils/combine_data.sh --extra-files utt2uniq ${data_dir}_multi_hires ${corrupted_data_dir} ${clean_data_dir}_clean_hires ${data_dir}_hires
####  fi
####fi
####
####if [ $stage -le 24 ]; then
####  local/snr/make_snr_targets.sh --length-tolerance 2 --compress false \
####    --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --target-type Irm --apply-exp true \
####    ${vad_scp:+--ali-rspecifier "scp:$vad_scp" --silence-phones-str "0:2:4:10"} \
####    ${clean_data_dir}_fbank ${noise_data_dir}_fbank ${corrupted_data_dir}_hires \
####    $tmpdir/$data_id $targets_dir || exit 1
####fi
####
####if $create_targets_for_uncorrupted_dir; then
####  if [ $stage -le 25 ]; then
####    utils/combine_data.sh --extra-files "irm_targets.scp" \
####      ${data_dir}_multi_hires ${corrupted_data_dir}_hires \
####      ${clean_data_dir}_clean_hires ${data_dir}_hires || exit 1
####    
####    utils/combine_data.sh --extra-files "irm_targets.scp" \
####      ${data_dir}_multi_fbank ${corrupted_data_dir}_fbank \
####      ${clean_data_dir}_clean_fbank ${data_dir}_fbank || exit 1
####  fi
####fi
####
####exit 0
####
