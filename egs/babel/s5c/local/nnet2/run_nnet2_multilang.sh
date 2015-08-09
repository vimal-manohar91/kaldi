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
dir=exp/multilang/tri6_nnet
create_egs=false
egs_root_dir=
do_decode=false
num_epochs=12
pnorm_input_dim=3000
pnorm_output_dim=300
num_hidden_layers=3
debug_mode=false
exit_stage=0
learning_rate_opts="--initial-learning-rate 0.008 --final-learning-rate 0.0008"

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

if [ $stage -le 7 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/${lang[0]}/egs/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/babel-$date/s5c/$dir/${lang[0]}/egs/storage ${dir}/${lang[0]}/egs/storage
  fi

  egs_dir=
  [ ! -z "$egs_root_dir" ] && egs_dir=$egs_root_dir/${lang[0]}/egs
  
  steps/nnet2/train_pnorm_simple2.sh \
    --stage $train_stage ${learning_rate_opts} \
    --num-hidden-layers $num_hidden_layers \
    --pnorm-input-dim $pnorm_input_dim \
    --pnorm-output-dim $pnorm_output_dim \
    "${dnn_gpu_parallel_opts[@]}" \
    --egs-dir "$egs_dir" \
    --cmd "$train_cmd" \
    --cleanup false --exit-stage $exit_stage \
    data/${dataid[0]} data/${lang[0]}/lang ${ali[0]} ${dir}/${lang[0]} || exit 1
fi

if $create_egs && [ $stage -le 8 ]; then
  for i in `seq 1 $nlangs`; do
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/${lang[$i]}/egs/storage ]; then
      date=$(date +'%m_%d_%H_%M')
      utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/babel-$date/s5c/$dir/${lang[$i]}/egs/storage ${dir}/${lang[$i]}/egs/storage
    fi

    steps/nnet2/get_egs2.sh \
      --io-opts "--max-jobs-run 10" --cmd "$train_cmd" \
      --transform-dir ${ali[$i]} \
      --left-context 14 --right-context 10 \
      data/${dataid[$i]} ${ali[$i]} $dir/${lang[$i]}/egs
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
    --num-epochs $num_epochs --mix-up "$mix_up" --max-jobs-run 11 \
    --initial-learning-rate 0.05 --final-learning-rate 0.005 \
    --cmd "$train_cmd" \
    $input_dirs $dir/${lang[0]}/10.mdl $dir
fi
