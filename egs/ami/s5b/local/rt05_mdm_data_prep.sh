#! /bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0.

. path.sh

cmd=run.pl
nj=4

. utils/parse_options.sh

set -e 
set -o pipefail
set -u

if [ $# -ne 2 ]; then
  echo "Usage: $0 <SOURCE-DIR> <mic>"
  echo " e.g.: $0 /export/corpora5/LDC/LDC2011S06 mdm"
  exit 1
fi

SOURCE_DIR=$1
mic=$2

dir=data/local/rt05_eval/$mic
logdir=data/local/beamforming/rt05_eval_${mic}
odir=data/local/beamformed/rt05_eval_${mic}   # TODO: Make it modifiable

mkdir -p $dir
cut -d ' ' -f 1 conf/channels_rt05_eval_${mic}_conf > $dir/meetings

splits=
for n in `seq $nj`; do
  splits="$splits $dir/meetings.$n"
done

utils/split_scp.pl $dir/meetings $splits

if [ ! -f $odir/.done ]; then
  $cmd JOB=1:$nj $logdir/beamformit.JOB.log \
    local/do_beamformit.sh $dir/meetings.JOB \
    conf/channels_rt05_eval_${mic}_conf \
    conf/rt05_eval_conf.cfg \
    $SOURCE_DIR/data/audio/eval05s/english/confmtg \
    $odir
  touch $odir/.done
fi

data=data/$mic/rt05_eval
mkdir -p $data

sph2pipe=`which sph2pipe` || exit 1 
for x in `find $odir -name "*.sph"`; do
  y=`basename $x .sph`
  echo "$y $sph2pipe -f wav $x |" 
done > $data/wav.scp

awk '{print $1" "$2}' $data/wav.scp > $data/utt2spk
cp $data/utt2spk $data/spk2utt
awk '{print $1" "$1" 1"}' $data/wav.scp > $data/reco2file_and_channel
