# you can change cmd.sh depending on what type of queue you are using.
# If you have no queueing system and want to run on a local machine, you
# can change all instances 'queue.pl' to run.pl (but be careful and run
# commands one by one: most recipes will exhaust the memory on your
# machine).  queue.pl works with GridEngine (qsub).  slurm.pl works
# with slurm.  Different queues are configured differently, with different
# queue names and different ways of specifying things like memory;
# to account for these differences you can create and edit the file
# conf/queue.conf to match your queue's configuration.  Search for
# conf/queue.conf in http://kaldi-asr.org/doc/queue.html for more information,
# or search for the string 'default_config' in utils/queue.pl or utils/slurm.pl.

#a) JHU cluster options
export train_cmd="queue.pl -l arch=*64*"
export decode_cmd="queue.pl -l arch=*64* --mem 4G"
#export cuda_cmd="..."
export mkgraph_cmd="queue.pl -l arch=*64* --mem 4G"

#b) BUT cluster options
#export train_cmd="queue.pl -q all.q@@blade -l ram_free=1200M,mem_free=1200M"
#export decode_cmd="queue.pl -q all.q@@blade -l ram_free=1700M,mem_free=1700M"
#export decodebig_cmd="queue.pl -q all.q@@blade -l ram_free=4G,mem_free=4G"
#export cuda_cmd="queue.pl -q long.q@@pco203 -l gpu=1"
#export cuda_cmd="queue.pl -q long.q@pcspeech-gpu"
#export mkgraph_cmd="queue.pl -q all.q@@servers -l ram_free=4G,mem_free=4G"

#c) run it locally...
#export train_cmd=run.pl
#export decode_cmd=run.pl
export cuda_cmd=run.pl
#export mkgraph_cmd=run.pl

