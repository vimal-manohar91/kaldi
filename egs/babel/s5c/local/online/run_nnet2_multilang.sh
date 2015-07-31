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
splice_indexes="layer0/-1:0:1 layer1/-2:1 layer2/-4:2"
do_decode=false

. path.sh
. cmd.sh

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

  steps/nnet2/train_multisplice_accel2.sh \
    --feat-type raw \
    --num-hidden-layers 4 \
    --splice-indexes "$splice_indexes" \
    --online-ivector-dir exp/nnet2_online/${lang[0]}/ivectors_train \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-jobs-initial 2 --num-jobs-final 8 \
    --num-epochs 15 \
    --initial-effective-lrate 0.005 --final-effective-lrate 0.0005 \
    --cmd "$train_cmd" \
    --pnorm-input-dim 2000 \
    --pnorm-output-dim 200 \
    --exit-stage 12 \
    $data_multilang/${lang[0]}_hires data/${lang[0]}/lang ${ali[0]} ${dir}/${lang[0]}
fi

if $create_egs && [ $stage -le 8 ]; then
  context_string=$(cat $dir/${lang[0]}/vars)
  echo $context_string
  eval $context_string || exit -1; #

  for i in `seq 1 $nlangs`; do
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/${lang[$i]}/egs/storage ]; then
      date=$(date +'%m_%d_%H_%M')
      utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/babel-$date/s5c/$dir/${lang[$i]}/egs/storage ${dir}/${lang[$i]}/egs/storage
    fi

    steps/nnet2/get_egs2.sh \
      --io-opts "--max-jobs-run 10" --cmd "$train_cmd" \
      --cmvn-opts "--norm-means=false --norm-vars=false" \
      --feat-type raw \
      --online-ivector-dir exp/nnet2_online/${lang[$i]}/ivectors_train \
      --transform-dir ${ali[$i]} \
      --left-context $nnet_left_context --right-context $nnet_right_context \
      $data_multilang/${lang[$i]}_hires ${ali[$i]} $dir/${lang[$i]}/egs
  done
fi


if [ $stage -le 9 ]; then
  #--num-hidden-layers 3 \
  #--splice-indexes "layer0/-2:-1:0:1:2 layer1/-3:1 layer2/-5:3" \

  input_dirs="${ali[0]} $dir/${lang[0]}/egs"
  num_jobs_nnet=1
  mix_up=0
  for n in `seq 1 $nlangs`; do
    input_dirs="$input_dirs ${ali[$n]} $dir/${lang[$n]}/egs"
    num_jobs_nnet="$num_jobs_nnet 1"
    mix_up="$mix_up 0"
  done

  steps/nnet2/train_multilang2.sh \
    --stage $train_stage \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-jobs-nnet "$num_jobs_nnet" \
    --num-epochs 10 --mix-up "$mix_up" --max-jobs-run 11 \
    --initial-learning-rate 0.05 --final-learning-rate 0.005 \
    --cmd "$train_cmd" \
    $input_dirs $dir/${lang[0]}/11.mdl $dir
fi

if [ $stage -le 10 ]; then
  for i in `seq 0 $[nlangs-1]`; do 
    steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
      data/${lang[$i]}/lang exp/nnet2_online/${lang[0]}/extractor $dir/$i ${dir}/${i}_online
  done
fi

if $do_decode; then
if [ $stage -le 11 ]; then
  for i in `seq 0 $[nlangs-1]`; do
    if [ ! -d exp/${lang[$i]}/tri5/graph ]; then
      utils/mkgraph.sh data/${lang[$i]}/lang exp/${lang[$i]}/tri5 exp/${lang[$i]}/tri5/graph
    fi
    steps/online/nnet2/decode.sh --config conf/decode.config \
      --cmd "$decode_cmd" --nj 12 \
      --bean $dnn_beam --lattice-beam $dnn_lat_beam --skip-scoring true \
      exp/${lang[$i]}/tri5/graph data/${lang[$i]}/dev2h \
      ${dir}/${i}_online/decode_dev2h || exit 1

    local/run_kws_stt_task.sh --max-states $max_states \
      --skip-scoring false --extra-kws false --wip $wip \
      --cmd "$decode_cmd" --skip-kws true --skip-stt false \
      "$lmwt_dnn_extra_opts[@]}" \
      data/${lang[$i]}/dev2h data/${lang[$i]}/lang ${dir}/${i}_online/decode_dev2h || exit 1

  done
fi
fi
