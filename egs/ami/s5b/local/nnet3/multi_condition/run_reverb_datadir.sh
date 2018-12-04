#!/bin/bash

set -e -o pipefail


# This script is called from local/chain/multi_condition/run_tdnn.sh.
# It contains the common feature preparation and iVector-related parts
# of the script.  See those scripts for examples of usage.

mic=ihm
nj=30
norvb_data_dir=data/ihm/train_cleaned_sp

rvb_affix=_rvb
num_data_reps=3
sample_rate=16000

max_jobs_run=10

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

echo "$0: creating reverberated MFCC features"

mfccdir=${norvb_data_dir}_rvb${num_data_reps}_hires/data
if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
  utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$mic-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
fi

rm -r ${norvb_data_dir}_rvb${num_data_reps} 2>/dev/null || true 

if [ ! -f ${norvb_data_dir}_rvb${num_data_reps}_hires/feats.scp ]; then
  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
  rvb_opts+=(--noise-set-parameters RIRS_NOISES/pointsource_noises/noise_list)

  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --prefix "rev" \
    --foreground-snrs "20:10:15:5:0" \
    --background-snrs "20:10:15:5:0" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 1 \
    --isotropic-noise-addition-probability 1 \
    --num-replications ${num_data_reps} \
    --max-noises-per-minute 1 \
    --source-sampling-rate $sample_rate \
    ${norvb_data_dir} ${norvb_data_dir}_rvb${num_data_reps}

  if [ -f $norvb_data_dir/utt2dur ]; then
    for n in $(seq $num_data_reps); do
      awk -v n=$n '{print "rev"n"_"$0}' $norvb_data_dir/utt2dur 
    done | sort -k1,1 > ${norvb_data_dir}_rvb${num_data_reps}/utt2dur
  fi

  utils/copy_data_dir.sh ${norvb_data_dir}_rvb${num_data_reps} ${norvb_data_dir}_rvb${num_data_reps}_hires

  # Volume perturbation has already been done
  #utils/data/perturb_data_dir_volume.sh ${norvb_data_dir}_rvb${num_data_reps}_hires

  steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
    --cmd "$train_cmd --max-jobs-run $max_jobs_run" ${norvb_data_dir}_rvb${num_data_reps}_hires
  steps/compute_cmvn_stats.sh ${norvb_data_dir}_rvb${num_data_reps}_hires
  utils/fix_data_dir.sh ${norvb_data_dir}_rvb${num_data_reps}_hires
fi

