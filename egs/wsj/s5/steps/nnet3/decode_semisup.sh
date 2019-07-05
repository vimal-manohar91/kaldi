#!/bin/bash

# Copyright 2012-2015  Johns Hopkins University (Author: Daniel Povey).
# Apache 2.0.

# This script does decoding with a neural-net.

# Begin configuration section.
stage=1
nj=4 # number of decoding jobs.
sub_split=1     # If decoding a large amount of data, then create small
                # subsplits of data to run smaller decoding jobs to avoid
                # large jobs failing.
keep_subsplit=false   # If true, then retain the lattices corresponding to subsplits.
                      # This will be useful if rescoring the lattice later.
                      # If false, then merge the lattices corresponding to the subsplits
                      # and remove the subsplit lattices.
acwt=0.1  # Just a default value, used for adaptation and beam-pruning..
post_decode_acwt=1.0  # can be used in 'chain' systems to scale acoustics by 10 so the
                      # regular scoring script works.
cmd=run.pl
beam=15.0
frames_per_chunk=50
max_active=7000
min_active=200
ivector_scale=1.0
lattice_beam=8.0 # Beam we use in lattice generation.
iter=final
num_threads=1 # if >1, will use gmm-latgen-faster-parallel
use_gpu=false # If true, will use a GPU, with nnet3-latgen-faster-batch.
              # In that case it is recommended to set num-threads to a large
              # number, e.g. 20 if you have that many free CPU slots on a GPU
              # node, and to use a small number of jobs.
scoring_opts=
skip_diagnostics=false
skip_scoring=false
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
online_ivector_dir=
minimize=false
word_determinize=false  # If set to true, then output lattice does not retain
                        # alternate paths a sequence of words (with alternate pronunciations).
                        # Setting to true is the default in steps/nnet3/decode.sh.
                        # However, setting this to false
                        # is useful for generation w of semi-supervised training
                        # supervision and frame-level confidences.
write_compact=true   # If set to false, then writes the lattice in non-compact format,
                     # retaining the acoustic scores on each arc. This is
                     # required to be false for LM rescoring undeterminized
                     # lattices (when --word-determinize is false)
                     # Useful for semi-supervised training with rescored lattices.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. utils/parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
  echo "e.g.:   steps/nnet3/decode.sh --nj 8 \\"
  echo "--online-ivector-dir exp/nnet2_online/ivectors_test_eval92 \\"
  echo "    exp/tri4b/graph_bg data/test_eval92_hires $dir/decode_bg_eval92"
  echo "main options (for others, see top of script file)"
  echo "  --transform-dir <decoding-dir>           # directory of previous decoding"
  echo "                                           # where we can find transforms for SAT systems."
  echo "  --config <config-file>                   # config containing options"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --beam <beam>                            # Decoding beam; default 15.0"
  echo "  --iter <iter>                            # Iteration of model to decode; default is final."
  echo "  --scoring-opts <string>                  # options to local/score.sh"
  echo "  --num-threads <n>                        # number of threads to use, default 1."
  exit 1;
fi

graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.
model=$srcdir/$iter.mdl


extra_files=
if [ ! -z "$online_ivector_dir" ]; then
  steps/nnet2/check_ivectors_compatible.sh $srcdir $online_ivector_dir || exit 1
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"
fi

utils/lang/check_phones_compatible.sh {$srcdir,$graphdir}/phones.txt || exit 1

for f in $graphdir/HCLG.fst $data/feats.scp $model $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

sdata=$data/split$nj;
cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1;
use_sliding_window_cmvn=false
if [ -f $srcdir/sliding_window_cmvn ]; then
  use_sliding_window_cmvn=`cat $srcdir/sliding_window_cmvn`
fi
thread_string=
if $use_gpu; then
  if [ $num_threads -eq 1 ]; then
    echo "$0: **Warning: we recommend to use --num-threads > 1 for GPU-based decoding."
  fi
  thread_string="-batch --num-threads=$num_threads"
  queue_opt="--num-threads $num_threads --gpu 1"
elif [ $num_threads -gt 1 ]; then
  thread_string="-parallel --num-threads=$num_threads"
  queue_opt="--num-threads $num_threads"
fi

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs


## Set up features.
echo "$0: feature type is raw"

if $use_sliding_window_cmvn; then
  feats="ark,s,cs:apply-cmvn-sliding $cmvn_opts scp:$sdata/JOB/feats.scp ark:- |"
else
  feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
fi

if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
fi

extra_opts=
lat_wspecifier="ark:|"
if ! $write_compact; then
  extra_opts="--determinize-lattice=false"
  lat_wspecifier="ark:| lattice-determinize-phone-pruned-parallel --num-threads=$num_threads --beam=$lattice_beam --acoustic-scale=$acwt --minimize=$minimize --word-determinize=$word_determinize --write-compact=false $model ark:- ark:- |"
fi

if [ "$post_decode_acwt" == 1.0 ]; then
  lat_wspecifier="$lat_wspecifier gzip -c >$dir/lat.JOB.gz"
else
  lat_wspecifier="$lat_wspecifier lattice-scale --acoustic-scale=$post_decode_acwt --write-compact=$write_compact ark:- ark:- | gzip -c >$dir/lat.JOB.gz"
fi

frame_subsampling_opt=
if [ -f $srcdir/frame_subsampling_factor ]; then
  # e.g. for 'chain' systems
  frame_subsampling_opt="--frame-subsampling-factor=$(cat $srcdir/frame_subsampling_factor)"
fi

# if this job is interrupted by the user, we want any background jobs to be
# killed too.
cleanup() {
  local pids=$(jobs -pr)
  [ -n "$pids" ] && kill $pids
}
trap "cleanup" INT QUIT TERM EXIT

# Copy the model as it is required when generating egs
cp $model $dir/final.mdl  || exit 1

if [ $stage -le 1 ]; then
  if [ $sub_split -eq 1 ]; then
    $cmd --num-threads $num_threads JOB=1:$nj $dir/log/decode.JOB.log \
      nnet3-latgen-faster$thread_string $ivector_opts $frame_subsampling_opt \
       --frames-per-chunk=$frames_per_chunk \
       --extra-left-context=$extra_left_context \
       --extra-right-context=$extra_right_context \
       --extra-left-context-initial=$extra_left_context_initial \
       --extra-right-context-final=$extra_right_context_final \
       --minimize=$minimize --word-determinize=$word_determinize \
       --max-active=$max_active --min-active=$min_active --beam=$beam \
       --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=true \
       --word-symbol-table=$graphdir/words.txt ${extra_opts} $model \
       $graphdir/HCLG.fst "$feats" "$lat_wspecifier" || exit 1;
  else
    # each job from 1 to $nj is split into multiple pieces (sub-split), and we aim
    # to have at most two jobs running at each time.  The idea is that if we have
    # stragglers from one job, we can be processing another one at the same time.
    rm $dir/.error 2>/dev/null

    prev_pid=
    for n in $(seq $[nj+1]); do
      lat_subset_wspecifier="ark:|"
      if ! $write_compact; then
        lat_subset_wspecifier="ark:| lattice-determinize-phone-pruned-parallel --num-threads=$num_threads --beam=$lattice_beam --acoustic-scale=$acwt --minimize=$minimize --word-determinize=$word_determinize --write-compact=false $model ark:- ark:- |"
      fi
      if [ "$post_decode_acwt" == 1.0 ]; then
        lat_subset_wspecifier="$lat_subset_wspecifier gzip -c >$dir/lat.$n.JOB.gz"
      else
        lat_subset_wspecifier="$lat_subset_wspecifier lattice-scale --acoustic-scale=$post_decode_acwt --write-compact=$write_compact ark:- ark:- | gzip -c >$dir/lat.$n.JOB.gz"
      fi

      if [ $n -gt $nj ]; then
        this_pid=
      elif [ -f $dir/.done.$n ] && [ $dir/.done.$n -nt $model ]; then
        echo "$0: Not processing subset $n as already done (delete $dir/.done.$n if not)";
        this_pid=
      else
        sdata2=$data/split$nj/$n/split${sub_split}utt;
        utils/split_data.sh --per-utt $sdata/$n $sub_split || exit 1;
        mkdir -p $dir/log/$n
        mkdir -p $dir/part
        feats_subset=$(echo $feats | sed s:JOB/:$n/split${sub_split}utt/JOB/:g)
        $cmd --num-threads $num_threads JOB=1:$sub_split $dir/log/$n/decode.JOB.log \
          nnet3-latgen-faster$thread_string $ivector_opts $frame_subsampling_opt \
           --frames-per-chunk=$frames_per_chunk \
           --extra-left-context=$extra_left_context \
           --extra-right-context=$extra_right_context \
           --extra-left-context-initial=$extra_left_context_initial \
           --extra-right-context-final=$extra_right_context_final \
           --minimize=$minimize --word-determinize=$word_determinize \
           --max-active=$max_active --min-active=$min_active --beam=$beam \
           --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=true \
           --word-symbol-table=$graphdir/words.txt ${extra_opts} $model \
           $graphdir/HCLG.fst "$feats_subset" "$lat_subset_wspecifier" || touch $dir/.error &
        this_pid=$!
      fi
      if [ ! -z "$prev_pid" ]; then # Wait for the previous job to merge lattices.
        wait $prev_pid
        [ -f $dir/.error ] && \
          echo "$0: error generating lattices" && exit 1;

        if ! $keep_subsplit; then
          rm $dir/.merge_error 2>/dev/null
          echo "$0: Merging archives for data subset $prev_n"
          for k in $(seq $sub_split); do
            gunzip -c $dir/lat.$prev_n.$k.gz || touch $dir/.merge_error;
          done | gzip -c > $dir/lat.$prev_n.gz || touch $dir/.merge_error;
          [ -f $dir/.merge_error ] && \
            echo "$0: Merging lattices for subset $prev_n failed" && exit 1;
          rm $dir/lat.$prev_n.*.gz
        fi
        touch $dir/.done.$prev_n
      fi
      prev_n=$n
      prev_pid=$this_pid
    done
  fi
fi

if $keep_subsplit; then
  echo $sub_split > $dir/sub_split
fi

if [ $stage -le 2 ]; then
  if ! $skip_diagnostics ; then
    [ ! -z $iter ] && iter_opt="--iter $iter"
    steps/diagnostic/analyze_lats.sh --cmd "$cmd" $iter_opt $graphdir $dir
  fi
fi


# The output of this script is the files "lat.*.gz"-- we'll rescore this at
# different acoustic scales to get the final output.
if [ $stage -le 3 ]; then
  if ! $skip_scoring ; then
    [ ! -x local/score.sh ] && \
      echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
    echo "score best paths"
    [ "$iter" != "final" ] && iter_opt="--iter $iter"
    local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
    echo "score confidence and timing with sclite"
  fi
fi
echo "Decoding done."
exit 0;
