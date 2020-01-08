export KALDI_ROOT=`pwd`/../../..
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH:/bin/:/opt/anaconda3/bin:/sbin
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$KALDI_ROOT/tools/tensorflow_build/.local/lib/python2.7/site-packages
export LC_ALL=C
. /etc/profile.d/modules.sh
module load shared cuda80/toolkit
