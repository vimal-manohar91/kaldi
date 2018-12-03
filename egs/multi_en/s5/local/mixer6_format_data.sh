#! /bin/bash

# Copyright 2018  Vimal Manohar
# Apache 2.0

. ./path.sh

only_good=true
no_unsuitable=true

. utils/parse_options.sh

local_dir=$1   # /export/corpora/LDC/LDC2013S03/mx6_speech
dir=$2

if [ $# -ne 2 ]; then
  echo "Usage: $0 <local-dir> <dir>"
  exit 1
fi

mkdir -p $dir

sph2pipe=$(which sph2pipe) || exit 1

cat $local_dir/wav_list | \
  python3 -c "
import sys
for line in sys.stdin:
  parts = line.strip().split()
  file_id = parts[0]
  file_path = parts[1]

  for channel in [1, 2]:
    reco_id = '{}-{}'.format(file_id, channel)

    print ('{reco_id} $sph2pipe -f wav -p -c {channel} {file_path} |'
           ''.format(reco_id=reco_id, channel=channel, file_path=file_path))
" > $dir/wav.scp

for f in reco2file_and_channel segments utt2spk; do 
  cp $local_dir/$f $dir/
done

if $only_good; then
  utils/filter_scp.pl $local_dir/good_utts \
    $local_dir/utt2spk > $dir/utt2spk
elif $no_unsuitable; then
  utils/filter_scp.pl --exclude $local_dir/unsuitable_utts \
    $local_dir/utt2spk > $dir/utt2spk
fi

utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

utils/fix_data_dir.sh $dir
