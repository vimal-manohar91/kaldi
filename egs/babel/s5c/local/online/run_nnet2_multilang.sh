# This example does multilingual online nnet2 training with i-vectors
# Eg: ./local/online/run_nnet2_multilang.sh --l=ASM --ali=exp/ASM/tri5_ali --data=data/ASM/train \
#											       --l=BNG --ali=exp/BNG/tri5_ali --data=data/BNG/train \
#												   --l=CNT --ali=exp/CNT/tri5_ali --data=data/CNT/train

set -e 
set -o pipefail
set -u

stage=-10
train_stage=-10
use_gpu=true
dir=exp/nnet2_online/nnet_ms_a_multilang
create_egs=false
egs_root_dir=
splice_indexes="layer0/-1:0:1 layer1/-2:1 layer2/-4:2"
do_decode=false
num_epochs=10
pnorm_input_dim=3000
pnorm_output_dim=300
num_hidden_layers=4
debug_mode=false
prepare_feats=false
exit_stage=10
learning_rate_opts="--initial-effective-lrate 0.005 --final-effective-lrate 0.0005"
multilang_learning_rate_opts="--initial-learning-rate 0.005 --final-learning-rate 0.0005"
use_no_ivec=false

. path.sh
. cmd.sh

. conf/common.limitedLP

. utils/parse_options.sh

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=512
  # the _a is in case I want to change the parameters.
else
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
fi

ARGS=$*
echo $ARGS

nlangs=0
j=0
while [ $# -gt 0 ]; do
  lang[j]=$1
  ali[j]=$2
  dataid[j]=$3

  shift; shift; shift;
  nlangs=$[nlangs+1]
  j=$nlangs
done

# Check if all the user i/p directories exist
nlangs=$[nlangs-1]
for i in  $(seq 0 $nlangs)
do
	echo "lang = ${lang[i]}, alidir = ${ali[i]}, dataid = ${dataid[i]}"
	[ ! -e ${ali[i]} ] && echo  "Missing  ${ali[i]}" && exit 1
	[ ! -e data/${dataid[i]} ] && echo "Missing data/${dataid[i]}" && exit 1
done

data_multilang=data_multi/train

local/online/run_nnet2_multilang_common.sh --stage $stage $ARGS

if [ $stage -le 7 ]; then
    #--num-hidden-layers 3 \
    #--splice-indexes "layer0/-2:-1:0:1:2 layer1/-3:1 layer2/-5:3" \

  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/${lang[0]}/egs/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/babel-$date/s5c/$dir/${lang[0]}/egs/storage ${dir}/${lang[0]}/egs/storage
  fi

  egs_dir=
  [ ! -z "$egs_root_dir" ] && egs_dir=$egs_root_dir/${lang[0]}/egs

  ivector_opts=
  cmvn_opts="--norm-means=true --norm-vars=false"
  ! $use_no_ivec && ivector_opts="--online-ivector-dir exp/nnet2_online/${lang[0]}/ivectors_train" && cmvn_opts="--norm-means=false --norm-vars=false"
  steps/nnet2/train_multisplice_accel2.sh \
    --stage $train_stage  --feat-type raw \
    --num-hidden-layers $num_hidden_layers \
    --splice-indexes "$splice_indexes" $ivector_opts \
    --cmvn-opts "$cmvn_opts" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-jobs-initial 2 --num-jobs-final 8 \
    --num-epochs 8 --egs-dir "$egs_dir" ${learning_rate_opts} \
    --cmd "$train_cmd" \
    --pnorm-input-dim $pnorm_input_dim \
    --pnorm-output-dim $pnorm_output_dim \
    --cleanup false --exit-stage $exit_stage \
    $data_multilang/${lang[0]}_hires data/${lang[0]}/lang ${ali[0]} ${dir}/${lang[0]}
fi

if $prepare_feats; then
  for i in `seq 0 $nlangs`; do 
    mfccdir=mfcc_hires/${lang[$i]}
    data_id=dev2h.pem
    
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
      date=$(date +'%m_%d_%H_%M')
      utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/babel-$date/s5c/$mfccdir/storage $mfccdir/storage
    fi

    utils/copy_data_dir.sh data/${lang[$i]}/$data_id data/${lang[$i]}/${data_id}_hires
    steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${lang[$i]}/${data_id}_hires exp/make_hires/$data_id $mfccdir || exit 1
    steps/compute_cmvn_stats.sh data/${lang[$i]}/${data_id}_hires exp/make_hires/$data_id $mfccdir || exit 1
  done
  
  for i in `seq 0 11`; do
    data_id=dev2h.pem
    this_data=data/${lang[$i]}/$data_id

    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" \
      --nj 12 \
      ${this_data}_hires exp/nnet2_online/${lang[0]}/extractor \
      exp/nnet2_online/${lang[$i]}/ivectors_${data_id} || exit 1
  done
fi

if $debug_mode; then
  if [ $stage -le 8 ]; then
    if [ ! -d exp/${lang[0]}/tri5/graph ]; then
      utils/mkgraph.sh data/${lang[0]}/lang exp/${lang[0]}/tri5 exp/${lang[0]}/tri5/graph
    fi
      
    ivector_opts=
    ! $use_no_ivec && ivector_opts="--online-ivector-dir exp/nnet2_online/${lang[0]}/ivectors_${data_id}"
    steps/nnet2/decode.sh --config conf/decode.config \
      --cmd "$decode_cmd" --nj 12 $ivector_opts \
      --beam $dnn_beam --lattice-beam $dnn_lat_beam --skip-scoring true \
      exp/${lang[0]}/tri5/graph data/${lang[0]}/dev2h.pem_hires \
      ${dir}/${lang[0]}/decode_dev2h.pem || exit 1

    local/run_kws_stt_task.sh --max-states $max_states \
      --skip-scoring false --extra-kws false --wip $wip \
      --cmd "$decode_cmd" --skip-kws true --skip-stt false \
      --min-lmwt 8 --max-lmwt 15 \
      data/${lang[0]}/dev2h.pem_hires data/${lang[0]}/lang ${dir}/${lang[0]}/decode_dev2h.pem || exit 1
  fi

  #if [ $stage -le 8 ]; then
  #  if [ ! -d exp/${lang[0]}/tri5/graph ]; then
  #    utils/mkgraph.sh data/${lang[0]}/lang exp/${lang[0]}/tri5 exp/${lang[0]}/tri5/graph
  #  fi
  #  ivector_opts=
  #  ! $use_no_ivec && ivector_opts=exp/nnet2_online/${lang[0]}/extractor
  #  steps/online/nnet2/prepare_online_decoding.sh --iter $exit_stage --mfcc-config conf/mfcc_hires.conf \
  #    data/${lang[$i]}/lang $ivector_opts $dir/${lang[0]} ${dir}/${lang[0]}_online

  #  steps/online/nnet2/decode.sh --config conf/decode.config \
  #    --cmd "$decode_cmd" --nj 12 \
  #    --beam $dnn_beam --lattice-beam $dnn_lat_beam --skip-scoring true \
  #    exp/${lang[0]}/tri5/graph data/${lang[0]}/dev2h.pem \
  #    ${dir}/${lang[0]}_online/decode_dev2h.pem || exit 1

  #  local/run_kws_stt_task.sh --max-states $max_states \
  #    --skip-scoring false --extra-kws false --wip $wip \
  #    --cmd "$decode_cmd" --skip-kws true --skip-stt false \
  #    --min-lmwt 8 --max-lmwt 15 \
  #    data/${lang[0]}/dev2h.pem data/${lang[0]}/lang ${dir}/${lang[0]}_online/decode_dev2h.pem || exit 1
  #fi
  echo "Exiting because --debug-mode is true" && exit 0 
fi

if $create_egs && [ $stage -le 8 ]; then
  context_string=$(cat $dir/${lang[0]}/vars)
  echo $context_string
  eval $context_string || exit -1; #

  nnet_left_context=19
  nnet_right_context=14

  for i in `seq 0 $nlangs`; do
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/${lang[$i]}/egs/storage ]; then
      date=$(date +'%m_%d_%H_%M')
      utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/babel-$date/s5c/$dir/${lang[$i]}/egs/storage ${dir}/${lang[$i]}/egs/storage
    fi

    ivector_opts=
    cmvn_opts="--norm-means=true --norm-vars=false"
    ! $use_no_ivec && ivector_opts="--online-ivector-dir exp/nnet2_online/${lang[i]}/ivectors_train" && cmvn_opts="--norm-means=false --norm-vars=false"
    steps/nnet2/get_egs2.sh \
      --io-opts "--max-jobs-run 10" --cmd "$train_cmd" \
      --cmvn-opts "$cmvn_opts" \
      --feat-type raw $ivector_opts \
      --transform-dir ${ali[$i]} \
      --left-context $nnet_left_context --right-context $nnet_right_context \
      $data_multilang/${lang[$i]}_hires ${ali[$i]} $dir/${lang[$i]}/egs
  done
fi

if [ $stage -le 9 ]; then
  #--num-hidden-layers 3 \
  #--splice-indexes "layer0/-2:-1:0:1:2 layer1/-3:1 layer2/-5:3" \
  
  egs_dir=$dir
  [ ! -z "$egs_root_dir" ] && egs_dir=$egs_root_dir

  input_dirs="${ali[0]} $egs_dir/${lang[0]}/egs"
  num_jobs_nnet=1
  mix_up=0
  for n in `seq 1 $nlangs`; do
    input_dirs="$input_dirs ${ali[$n]} $egs_dir/${lang[$n]}/egs"
    num_jobs_nnet="$num_jobs_nnet 1"
    mix_up="$mix_up 0"
  done

  steps/nnet2/train_multilang2.sh \
    --stage $train_stage \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-jobs-nnet "$num_jobs_nnet" \
    --num-epochs $num_epochs --mix-up "$mix_up" --max-jobs-run 11 \
    --cmd "$train_cmd" $multilang_learning_rate_opts \
    $input_dirs $dir/${lang[0]}/$exit_stage.mdl $dir
fi

if [ $stage -le 10 ]; then
  for i in `seq 0 $[nlangs-1]`; do 
    ivector_opts=
    ! $use_no_ivec && ivector_opts="exp/nnet2_online/${lang[0]}/extractor"
    steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
      data/${lang[$i]}/lang $ivector_opts $dir/$i ${dir}/${i}_online
  done
fi

if $do_decode; then
if [ $stage -le 11 ]; then
  for i in `seq 0 $[nlangs-1]`; do
    if [ ! -d exp/${lang[$i]}/tri5/graph ]; then
      utils/mkgraph.sh data/${lang[$i]}/lang exp/${lang[$i]}/tri5 exp/${lang[$i]}/tri5/graph
    fi
    ivector_opts=
    ! $use_no_ivec && ivector_opts="--online-ivector-dir exp/nnet2_online/${lang[$i]}/ivectors_${data_id}"
    steps/nnet2/decode.sh --config conf/decode.config \
      --cmd "$decode_cmd" --nj 12 $ivector_opts \
      --beam $dnn_beam --lattice-beam $dnn_lat_beam --skip-scoring true \
      exp/${lang[$i]}/tri5/graph data/${lang[$i]}/dev2h.pem_hires \
      ${dir}/${i}/decode_dev2h.pem || exit 1

    local/run_kws_stt_task.sh --max-states $max_states \
      --skip-scoring false --extra-kws false --wip $wip \
      --cmd "$decode_cmd" --skip-kws true --skip-stt false \
      --min-lmwt 8 --max-lmwt 15 \
      data/${lang[$i]}/dev2h.pem_hires data/${lang[$i]}/lang ${dir}/${i}/decode_dev2h.pem || exit 1
  done
fi

#if [ $stage -le 11 ]; then
#  for i in `seq 0 $[nlangs-1]`; do
#    if [ ! -d exp/${lang[$i]}/tri5/graph ]; then
#      utils/mkgraph.sh data/${lang[$i]}/lang exp/${lang[$i]}/tri5 exp/${lang[$i]}/tri5/graph
#    fi
#    steps/online/nnet2/decode.sh --config conf/decode.config \
#      --cmd "$decode_cmd" --nj 12 \
#      --beam $dnn_beam --lattice-beam $dnn_lat_beam --skip-scoring true \
#      exp/${lang[$i]}/tri5/graph data/${lang[$i]}/dev2h.pem \
#      ${dir}/${i}_online/decode_dev2h.pem || exit 1
#
#    local/run_kws_stt_task.sh --max-states $max_states \
#      --skip-scoring false --extra-kws false --wip $wip \
#      --cmd "$decode_cmd" --skip-kws true --skip-stt false \
#      --min-lmwt 8 --max-lmwt 15 \
#      data/${lang[$i]}/dev2h.pem data/${lang[$i]}/lang ${dir}/${i}_online/decode_dev2h.pem || exit 1
#
#  done
#fi
fi
