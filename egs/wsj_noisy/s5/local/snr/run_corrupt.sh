#!/bin/bash
set -e
set -o pipefail

. path.sh
. cmd.sh

num_data_reps=5
data_dir=data/train_si284
dest_wav_dir=wavs
nj=40
stage=1
corruption_stage=-10
pad_silence=false
mfcc_config=conf/mfcc_hires.conf
fbank_config=conf/fbank.conf
data_only=true
corrupt_only=true
dry_run=true
speed_perturb=false
vad_scp=
uncorrupted_vad_scp=
max_jobs_run=20

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  exit 1
fi

dataid=`basename ${data_dir}`

if [ $stage -le $num_data_reps ]; then
  corrupted_data_dirs=
  start_state=1
  if [ $stage -gt 1 ]; then 
    start_stage=$stage
  fi
  for x in `seq $start_stage $num_data_reps`; do
    cur_dest_dir=data/temp_${dataid}_$x
    output_clean_dir=data/temp_clean_${dataid}_$x
    output_noise_dir=data/temp_noise_${dataid}_$x
    local/snr/corrupt_data_dir.sh --dry-run $dry_run --random-seed $x --dest-wav-dir $dest_wav_dir/corrupted$x \
      --output-clean-wav-dir $dest_wav_dir/clean$x --output-clean-dir $output_clean_dir \
      --output-noise-wav-dir $dest_wav_dir/noise$x --output-noise-dir $output_noise_dir \
      --pad-silence $pad_silence --stage $corruption_stage --tmp-dir exp/make_corrupt_$dataid/$x \
      --nj $nj $data_dir data/impulse_noises $cur_dest_dir
    corrupted_data_dirs+=" $cur_dest_dir"
    clean_data_dirs+=" $output_clean_dir"
    noise_data_dirs+=" $output_noise_dir"
  done

  rm -r ${data_dir}_{corrupted,clean,noise} || true
  utils/combine_data.sh --extra-files utt2uniq ${data_dir}_corrupted ${corrupted_data_dirs}
  utils/combine_data.sh --extra-files utt2uniq ${data_dir}_clean ${clean_data_dirs}
  utils/combine_data.sh --extra-files utt2uniq ${data_dir}_noise ${noise_data_dirs}
  
  rm -r $corrupted_data_dirs || true
  rm -r $clean_data_dirs || true
fi

if [ ! -z "$uncorrupted_vad_scp" ]; then
  vad_scp=${data_dir}_corrupted/vad_multi.scp
  for x in `seq 1 $num_data_reps`; do
    awk -v x=$x '{print "corrupted"x"_"$0}' $uncorrupted_vad_scp > ${data_dir}_corrupted/vad_corrupted.scp
    awk -v x=$x '{print "clean-"$0}' ${data_dir}_corrupted/vad_corrupted.scp 
  done | cat - $uncorrupted_vad_scp | sort -k1,1 > ${data_dir}_corrupted/vad_multi.scp
fi

if $speed_perturb; then
  if [ ! -z "$uncorrupted_vad_scp" ]; then
    vad_scp=${data_dir}_corrupted/vad_multi_sp.scp
    for x in 0.9 1.0 1.1; do
      awk -v x=$x '{print "sp"x"-"$0}' $vad_scp > ${data_dir}_corrupted/vad_multi_sp.scp
    done
  fi

  if [ $stage -le 11 ]; then
    utils/data/perturb_data_dir_speed_3way.sh ${data_dir}_corrupted ${data_dir}_sp_corrupted
    utils/data/perturb_data_dir_speed_3way.sh ${data_dir}_clean ${data_dir}_sp_clean
    utils/data/perturb_data_dir_speed_3way.sh ${data_dir}_noise ${data_dir}_sp_noise
    utils/data/perturb_data_dir_speed_3way.sh ${data_dir} ${data_dir}_sp
  fi
  data_dir=${data_dir}_sp
fi

data_id=`basename $data_dir`
corrupted_data_dir=${data_dir}_corrupted
corrupted_data_id=`basename $corrupted_data_dir`
clean_data_dir=${data_dir}_clean
clean_data_id=`basename $clean_data_dir`
noise_data_dir=${data_dir}_noise
noise_data_id=`basename $noise_data_dir`

$corrupt_only && echo "--corrupt-only is true" && exit 1

mfccdir=mfcc_hires
#if [ $stage -le 2 ]; then
#  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
#    date=$(date +'%m_%d_%H_%M')
#    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$mfccdir/storage $mfccdir/storage
#  fi
#
#  utils/copy_data_dir.sh ${clean_data_dir} ${clean_data_dir}_hires
#  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config $mfcc_config ${clean_data_dir}_hires exp/make_hires/${clean_data_id} mfcc_hires
#  steps/compute_cmvn_stats.sh ${clean_data_dir}_hires exp/make_hires/${clean_data_id} mfcc_hires
#  utils/fix_data_dir.sh ${clean_data_dir}_hires
#fi

fbankdir=fbank_feats
if [ $stage -le 12 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $fbankdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$fbankdir/storage $fbankdir/storage
  fi

  rm -r ${clean_data_dir}_fbank || true
  utils/copy_data_dir.sh ${clean_data_dir} ${clean_data_dir}_fbank
  steps/make_fbank.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --fbank-config $fbank_config ${clean_data_dir}_fbank exp/make_fbank/${clean_data_id} fbank_feats 
  steps/compute_cmvn_stats.sh --fake ${clean_data_dir}_fbank exp/make_fbank/${clean_data_id} fbank_feats
  utils/fix_data_dir.sh ${clean_data_dir}_fbank
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $fbankdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$fbankdir/storage $fbankdir/storage
  fi

  rm -r ${noise_data_dir}_fbank || true
  utils/copy_data_dir.sh ${noise_data_dir} ${noise_data_dir}_fbank
  steps/make_fbank.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --fbank-config $fbank_config ${noise_data_dir}_fbank exp/make_fbank/${noise_data_id} fbank_feats 
  steps/compute_cmvn_stats.sh --fake ${noise_data_dir}_fbank exp/make_fbank/${noise_data_id} fbank_feats
  utils/fix_data_dir.sh ${noise_data_dir}_fbank
fi

if [ $stage -le 14 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $fbankdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$fbankdir/storage $fbankdir/storage
  fi
  
  rm -r ${corrupted_data_dir}_fbank || true
  utils/copy_data_dir.sh ${corrupted_data_dir} ${corrupted_data_dir}_fbank
  steps/make_fbank.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --fbank-config $fbank_config ${corrupted_data_dir}_fbank exp/make_fbank/${corrupted_data_id} fbank_feats 
  steps/compute_cmvn_stats.sh --fake ${corrupted_data_dir}_fbank exp/make_fbank/${corrupted_data_id} fbank_feats
  utils/fix_data_dir.sh --utt-extra-files utt2uniq ${corrupted_data_dir}_fbank
fi

if [ $stage -le 15 ]; then
  rm -r ${data_dir}_fbank || true
  utils/copy_data_dir.sh ${data_dir} ${data_dir}_fbank
  steps/make_fbank.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --fbank-config $fbank_config ${data_dir}_fbank exp/make_fbank/${data_id} fbank_feats 
  steps/compute_cmvn_stats.sh --fake ${data_dir}_fbank exp/make_fbank/${data_id} fbank_feats
  utils/fix_data_dir.sh --utt-extra-files utt2uniq ${data_dir}_fbank
fi

if [ $stage -le 16 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$mfccdir/storage $mfccdir/storage
  fi

  rm -r ${corrupted_data_dir}_hires || true
  utils/copy_data_dir.sh ${corrupted_data_dir} ${corrupted_data_dir}_hires
  steps/make_mfcc.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --mfcc-config $mfcc_config ${corrupted_data_dir}_hires exp/make_hires/${corrupted_data_id} mfcc_hires
  steps/compute_cmvn_stats.sh --fake ${corrupted_data_dir}_hires exp/make_hires/${corrupted_data_id} mfcc_hires
  utils/fix_data_dir.sh --utt-extra-files utt2uniq ${corrupted_data_dir}_hires
fi

if [ $stage -le 17 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$mfccdir/storage $mfccdir/storage
  fi

  rm -r ${clean_data_dir}_hires || true
  utils/copy_data_dir.sh ${clean_data_dir} ${clean_data_dir}_hires
  steps/make_mfcc.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --mfcc-config $mfcc_config ${clean_data_dir}_hires exp/make_hires/${clean_data_id} mfcc_hires 
  steps/compute_cmvn_stats.sh --fake ${clean_data_dir}_hires exp/make_hires/${clean_data_id} mfcc_hires
  utils/fix_data_dir.sh --utt-extra-files utt2uniq ${clean_data_dir}_hires
fi

if [ $stage -le 18 ]; then
  utils/copy_data_dir.sh --utt-prefix "clean-" --spk-prefix "clean-" ${clean_data_dir}_fbank ${clean_data_dir}_clean_fbank
  utils/copy_data_dir.sh --utt-prefix "clean-" --spk-prefix "clean-" ${clean_data_dir}_hires ${clean_data_dir}_clean_hires
fi

if [ $stage -le 19 ]; then
  rm -r ${data_dir}_hires || true
  utils/copy_data_dir.sh ${data_dir} ${data_dir}_hires
  steps/make_mfcc.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --mfcc-config $mfcc_config ${data_dir}_hires exp/make_hires/${data_id} mfcc_hires 
  steps/compute_cmvn_stats.sh --fake ${data_dir}_hires exp/make_hires/${data_id} mfcc_hires
  utils/fix_data_dir.sh --utt-extra-files utt2uniq ${data_dir}_hires
fi

if [ $stage -le 20 ]; then
utils/combine_data.sh --extra-files utt2uniq ${data_dir}_multi_fbank ${corrupted_data_dir} ${clean_data_dir}_clean_fbank ${data_dir}_fbank
utils/combine_data.sh --extra-files utt2uniq ${data_dir}_multi_hires ${corrupted_data_dir} ${clean_data_dir}_clean_hires ${data_dir}_hires
fi

[ $(cat ${clean_data_dir}_fbank/utt2spk | wc -l) -ne $(cat ${corrupted_data_dir}_fbank/utt2spk | wc -l) ] && echo "$0: ${clean_data_dir}_fbank/utt2spk and ${corrupted_data_dir}_fbank/utt2spk have different number of lines" && exit 1

[ $(cat ${noise_data_dir}_fbank/utt2spk | wc -l) -ne $(cat ${corrupted_data_dir}_fbank/utt2spk | wc -l) ] && echo "$0: ${noise_data_dir}_fbank/utt2spk and ${corrupted_data_dir}_fbank/utt2spk have different number of lines" && exit 1

$data_only && echo "--data-only is true" && exit 1

tmpdir=exp/make_irm_targets
targets_dir=irm_targets
  
if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $targets_dir/storage ]; then
  date=$(date +'%m_%d_%H_%M')
  utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$targets_dir/storage $targets_dir/storage
  for n in `seq $nj`; do 
    utils/create_data_link.pl $targets_dir/${data_id}.$n.ark
  done
fi

if [ $stage -le 21 ]; then
  local/snr/make_snr_targets.sh --length-tolerance 2 --compress false \
    --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --target-type Irm --apply-exp true \
    ${vad_scp:+--ali-rspecifier "scp:$vad_scp" --silence-phones-str "0:2:4:10"} \
    --ignore-noise-dir true \
    ${clean_data_dir}_fbank ${noise_data_dir}_fbank ${data_dir}_hires \
    $tmpdir/$data_id $targets_dir || exit 1
fi

if [ $stage -le 22 ]; then
  utils/copy_data_dir.sh --utt-prefix clean- --spk-prefix clean- --extra-files "irm_targets.scp" \
    ${clean_data_dir}_hires ${clean_data_dir}_clean_hires || exit 1

  local/snr/make_snr_targets.sh --length-tolerance 2 --compress false \
    --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --target-type Irm --apply-exp true \
    ${vad_scp:+--ali-rspecifier "scp:$vad_scp" --silence-phones-str "0:2:4:10"} \
    --ignore-noise-dir true \
    ${clean_data_dir}_clean_fbank ${noise_data_dir}_fbank ${clean_data_dir}_clean_hires \
    $tmpdir/$data_id $targets_dir || exit 1
fi

if [ $stage -le 23 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/wsj_noisy-$date/s5/$mfccdir/storage $mfccdir/storage
  fi

  rm -r ${corrupted_data_dir}_hires || true
  utils/copy_data_dir.sh ${corrupted_data_dir} ${corrupted_data_dir}_hires
  steps/make_mfcc.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --mfcc-config $mfcc_config ${corrupted_data_dir}_hires exp/make_hires/${corrupted_data_id} mfcc_hires
  steps/compute_cmvn_stats.sh --fake ${corrupted_data_dir}_hires exp/make_hires/${corrupted_data_id} mfcc_hires
  utils/fix_data_dir.sh --utt-extra-files utt2uniq ${corrupted_data_dir}_hires
fi

if [ $stage -le 23 ]; then
utils/combine_data.sh --extra-files utt2uniq ${data_dir}_multi_fbank ${corrupted_data_dir} ${clean_data_dir}_clean_fbank ${data_dir}_fbank
utils/combine_data.sh --extra-files utt2uniq ${data_dir}_multi_hires ${corrupted_data_dir} ${clean_data_dir}_clean_hires ${data_dir}_hires
fi


if [ $stage -le 23 ]; then
  local/snr/make_snr_targets.sh --length-tolerance 2 --compress false \
    --cmd "$train_cmd --max-jobs-run $max_jobs_run" --nj $nj --target-type Irm --apply-exp true \
    ${vad_scp:+--ali-rspecifier "scp:$vad_scp" --silence-phones-str "0:2:4:10"} \
    ${clean_data_dir}_fbank ${noise_data_dir}_fbank ${corrupted_data_dir}_hires \
    $tmpdir/$data_id $targets_dir || exit 1
fi

if [ $stage -le 24 ]; then
  utils/combine_data.sh --extra-files "irm_targets.scp" \
    ${data_dir}_multi_hires ${corrupted_data_dir}_hires \
    ${clean_data_dir}_clean_hires ${data_dir}_hires || exit 1
  
  utils/combine_data.sh --extra-files "irm_targets.scp" \
    ${data_dir}_multi_fbank ${corrupted_data_dir}_fbank \
    ${clean_data_dir}_clean_fbank ${data_dir}_fbank || exit 1
fi

exit 0

