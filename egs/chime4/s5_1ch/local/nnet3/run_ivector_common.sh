#!/bin/bash

set -e
stage=1
train_stage=-10
generate_alignments=true # false if doing ctc training
nj=30

#chime4 specific options
train=noisy

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s <enhancement method>\n\n" `basename $0`
  echo "First argument specifies a unique name for different enhancement method"
  exit 1;
fi

# set enhanced data
enhan=$1

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# check whether run_init is executed
if [ ! -d data/lang ]; then
  echo "error, execute local/run_init.sh, first"
  exit 1;
fi

# check whether run_init is executed
if [ ! -d exp/tri3b_tr05_multi_${train} ]; then
  echo "error, execute local/run_init.sh, first"
  exit 1;
fi

nnet3_dir=nnet3_$enhan
mkdir -p $nnet3_dir

# perturbed data preparation
if [ $stage -le 1 ]; then
  #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
  # _sp stands for speed-perturbed

  for datadir in tr05_multi_${train}; do
    utils/perturb_data_dir_speed.sh 0.9 data/${datadir} data/temp1
    utils/perturb_data_dir_speed.sh 1.1 data/${datadir} data/temp2
    utils/combine_data.sh data/${datadir}_tmp data/temp1 data/temp2
    utils/validate_data_dir.sh --no-feats data/${datadir}_tmp
    rm -r data/temp1 data/temp2

    mfccdir=mfcc_perturbed
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
      data/${datadir}_tmp exp/make_mfcc/${datadir}_tmp $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_tmp exp/make_mfcc/${datadir}_tmp $mfccdir || exit 1;
    utils/fix_data_dir.sh data/${datadir}_tmp

    utils/copy_data_dir.sh --spk-prefix sp1.0- --utt-prefix sp1.0- data/${datadir} data/temp0
    utils/combine_data.sh data/${datadir}_sp data/${datadir}_tmp data/temp0
    utils/fix_data_dir.sh data/${datadir}_sp
    rm -r data/temp0 data/${datadir}_tmp
  done
fi

train_set=tr05_multi_${train}_sp
if [ $stage -le 2 ] && [ "$generate_alignments" == "true" ]; then
  #obtain the alignment of the perturbed data
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/$train_set data/lang exp/tri3b_tr05_multi_${train} exp/tri3b_${train_set}_ali || exit 1
fi

if [ $stage -le 3 ]; then
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/chime4-$date/s5b/$mfccdir/storage $mfccdir/storage
  fi

  for dataset in $train_set; do
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires

    # scale the waveforms, this is useful as we don't use CMVN
    data_dir=data/${dataset}_hires
    cat $data_dir/wav.scp | python -c "
import sys, os, subprocess, re, random
random.seed(0)
scale_low = 1.0/8
scale_high = 2.0
for line in sys.stdin.readlines():
  if len(line.strip()) == 0:
    continue
  if line.strip()[-1] == '|':
    print '{0} sox --vol {1} -t wav - -t wav - |'.format(line.strip(), random.uniform(scale_low, scale_high))
  else:
    parts = line.split()
    print '{id} sox --vol {vol} -t wav {wav} -t wav - |'.format(id = parts[0], wav=' '.join(parts[1:]), vol = random.uniform(scale_low, scale_high))
"| sort -k1,1 -u  > $data_dir/wav.scp_scaled || exit 1;
    mv $data_dir/wav.scp_scaled $data_dir/wav.scp

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}_hires;
  done

  for dataset in dt05_real_$enhan dt05_simu_$enhan; do
    # Create MFCCs for the eval set
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 4 --mfcc-config conf/mfcc_hires.conf \
        data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires  # remove segments with problems
  done
fi

# ivector extractor training
if [ $stage -le 4 ]; then
  # We need to build the LDA+MLLT transform
  # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
  # the transform (12th iter is the last), any further training is pointless.
  # this decision is based on fisher_english
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --splice-opts "--left-context=3 --right-context=3" \
    5500 90000 data/${train_set}_hires \
    data/lang exp/tri3b_${train_set}_ali exp/$nnet3_dir/tri4a
fi

if [ $stage -le 5 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 200000 \
    data/${train_set}_hires 512 exp/$nnet3_dir/tri4a exp/$nnet3_dir/diag_ubm
fi

if [ $stage -le 6 ]; then
  # iVector extractors can be sensitive to the amount of data, but this one has a
  # fairly small dim (defaults to 100) so we don't use all of it, we use just the
  # 100k subset (just under half the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/${train_set}_hires exp/$nnet3_dir/diag_ubm exp/$nnet3_dir/extractor || exit 1;
fi

if [ $stage -le 7 ]; then
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero) so we
  # create artificial speakers with 2 utterances per speaker
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/${train_set}_hires data/${train_set}_max2_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    data/${train_set}_max2_hires exp/$nnet3_dir/extractor exp/$nnet3_dir/ivectors_$train_set || exit 1;

  for data_set in dt05_real_$enhan dt05_simu_$enhan; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 4 \
      data/${data_set}_hires exp/$nnet3_dir/extractor exp/$nnet3_dir/ivectors_$data_set || exit 1;
  done
fi

exit 0;
