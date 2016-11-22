#!/usr/bin/env python


# Copyright 2016 Vijayaditya Peddinti.
# Apache 2.0.


# this script is based on steps/nnet3/chain/train.sh

import os
import subprocess
import argparse
import sys
import pprint
import logging
import imp
import traceback
import shutil
import math

common_train_lib = imp.load_source('ntl', 'steps/nnet3/libs/common_train_lib.py')
chain_lib = imp.load_source('ncl', 'steps/nnet3/libs/chain_train_lib.py')
nnet3_log_parse = imp.load_source('nlp', 'steps/nnet3/report/nnet3_log_parse_lib.py')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting chain model trainer (train.py)')


def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="""
    Trains RNN and DNN acoustic models using the 'chain' objective function.
    """,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    conflict_handler = 'resolve')

    common_train_lib.AddCommonTrainArgs(parser)

    # egs extraction options
    parser.add_argument("--egs.chunk-width", type=int, dest='chunk_width',
                        default = 150,
                        help="Number of output labels in each example. Caution: if you double this you should halve --trainer.samples-per-iter.")

    # chain options
    parser.add_argument("--chain.lm-opts", type=str, dest='lm_opts',
                        default = None, action = common_train_lib.NullstrToNoneAction,
                        help="options to be be passed to chain-est-phone-lm")
    parser.add_argument("--chain.l2-regularize", type=float, dest='l2_regularize',
                        default = 0.0,
                        help="Weight of regularization function which is the"
                        " l2-norm of the output of the network. It should be"
                        " used without the log-softmax layer for the outputs."
                        " As l2-norm of the log-softmax outputs can dominate"
                        " the objective function.")
    parser.add_argument("--chain.xent-regularize", type=float, dest='xent_regularize',
                        default = 0.0,
                        help="Weight of regularization function which is the"
                        " cross-entropy cost the outputs.")
    parser.add_argument("--chain.right-tolerance", type=int, dest='right_tolerance',
                        default = 5, help="")
    parser.add_argument("--chain.left-tolerance", type=int, dest='left_tolerance',
                        default = 5, help="")
    parser.add_argument("--chain.leaky-hmm-coefficient", type=float, dest='leaky_hmm_coefficient',
                        default = 0.00001, help="")
    parser.add_argument("--chain.apply-deriv-weights", type=str, dest='apply_deriv_weights',
                        default=True, action=common_train_lib.StrToBoolAction,
                        choices = ["true", "false"],
                        help="")
    parser.add_argument("--chain.truncate-deriv-weights", type=float, dest='truncate_deriv_weights',
                        default =0,
                        help="Can be used to set to zero the weights of derivs"
                        " from frames near the edges.  (counts subsampled frames)")
    parser.add_argument("--chain.frame-subsampling-factor", type=int,
                        dest='frame_subsampling_factor',
                        default = 3,
                        help="ratio of frames-per-second of features we train"
                        " on, to chain model's output")
    parser.add_argument("--chain.alignment-subsampling-factor", type=int,
                        dest='alignment_subsampling_factor',
                        default = 3,
                        help="ratio of frames-per-second of input alignments to"
                        " chain model's output")
    parser.add_argument("--chain.left-deriv-truncate", type=int,
                        dest='left_deriv_truncate',
                        default = None, help="Deprecated. Kept for back compatibility")

    # trainer options
    parser.add_argument("--trainer.num-epochs", type=int, dest='num_epochs',
                        default = 10,
                        help="Number of epochs to train the model")
    parser.add_argument("--trainer.frames-per-iter", type=int, dest='frames_per_iter',
                        default=800000,
                        help ="Each iteration of training, see this many [input]"
                        " frames per job.  This option is passed to get_egs.sh."
                        " Aim for about a minute of training time")

    # Parameters for the optimization
    parser.add_argument("--trainer.optimization.initial-effective-lrate", type=float, dest='initial_effective_lrate',
                        default = 0.0002,
                        help="Learning rate used during the initial iteration")
    parser.add_argument("--trainer.optimization.final-effective-lrate", type=float, dest='final_effective_lrate',
                        default = 0.00002,
                        help="Learning rate used during the final iteration")
    parser.add_argument("--trainer.optimization.shrink-value", type=float, dest='shrink_value',
                        default = 1.0,
                        help="Scaling factor used for scaling the parameter"
                        " matrices when the derivative averages are below the"
                        " shrink-threshold at the non-linearities")
    parser.add_argument("--trainer.optimization.shrink-threshold", type=float, dest='shrink_threshold',
                        default = 0.15,
                        help="If the derivative averages are below this"
                        " threshold we scale the parameter matrices with the"
                        " shrink-value. It is less than 0.25 for sigmoid non-linearities.")
    parser.add_argument("--trainer.optimization.shrink-nonlinearity", type=str, dest='shrink_nonlinearity',
                        default = "SigmoidComponent", choices = ["TanhComponent", "SigmoidComponent"],
                        help="The non-linear component from which the"
                        " deriv-avg values are going to used to compute"
                        " mean-deriv-avg. The mean-deriv-avg is going to be"
                        " compared with shrink-threshold. Be careful to specify"
                        " a shrink-threshold which is dependent on the"
                        " shrink-nonlinearity type")

    # RNN specific trainer options
    parser.add_argument("--trainer.num-chunk-per-minibatch", type=int, dest='num_chunk_per_minibatch',
                        default=512,
                        help="Number of sequences to be processed in parallel every minibatch" )
    parser.add_argument("--trainer.deriv-truncate-margin", type=int, dest='deriv_truncate_margin',
                        default = None,
                        help="If specified, it is the number of frames that the derivative will be backpropagated through the chunk boundaries, "
                        "e.g., During BLSTM model training if the chunk-width=150 and deriv-truncate-margin=5, then the derivative will be "
                        "backpropagated up to t=-5 and t=154 in the forward and backward LSTM sequence respectively; "
                        "otherwise, the derivative will be backpropagated to the end of the sequence.")

    # General options
    parser.add_argument("--feat-dir", type=str, required = True,
                        help="Directory with features used for training the neural network.")
    parser.add_argument("--tree-dir", type=str, required = True,
                        help="Directory containing the tree to use for this model (we also "
                        "expect final.mdl and ali.*.gz in that directory")
    parser.add_argument("--lat-dir", type=str, required = True,
                        help="Directory with alignments used for training the neural network.")
    parser.add_argument("--dir", type=str, required = True,
                        help="Directory to store the models and all other files.")

    print(' '.join(sys.argv))
    print(sys.argv)

    args = parser.parse_args()

    [args, run_opts] = ProcessArgs(args)

    return [args, run_opts]

def ProcessArgs(args):
    # process the options
    if args.chunk_width < 1:
        raise Exception("--egs.chunk-width should have a minimum value of 1")

    if args.chunk_left_context < 0:
        raise Exception("--egs.chunk-left-context should be non-negative")

    if args.chunk_right_context < 0:
        raise Exception("--egs.chunk-right-context should be non-negative")

    if not args.left_deriv_truncate is None:
        args.deriv_truncate_margin = -args.left_deriv_truncate
        logger.warning("--chain.left-deriv-truncate (deprecated) is set by user, "
                "and --trainer.deriv-truncate-margin is set to negative of that value={0}. "
                "We recommend using the option --trainer.deriv-truncate-margin.".format(args.deriv_truncate_margin))

    if (not os.path.exists(args.dir)) or (not os.path.exists(args.dir+"/configs")):
        raise Exception("""This scripts expects {0} to exist and have a configs
        directory which is the output of make_configs.py script""")

    if args.transform_dir is None:
        args.transform_dir = args.lat_dir
    # set the options corresponding to args.use_gpu
    run_opts = common_train_lib.RunOpts()
    if args.use_gpu:
        if not common_train_lib.CheckIfCudaCompiled():
            logger.warning("""
    You are running with one thread but you have not compiled
    for CUDA.  You may be running a setup optimized for GPUs.  If you have
    GPUs and have nvcc installed, go to src/ and do ./configure; make""")

        run_opts.train_queue_opt = "--gpu 1"
        run_opts.parallel_train_opts = ""
        run_opts.combine_queue_opt = "--gpu 1"

    else:
        logger.warning("""
    Without using a GPU this will be very slow.  nnet3 does not yet support multiple threads.""")

        run_opts.train_queue_opt = ""
        run_opts.parallel_train_opts = "--use-gpu=no"
        run_opts.combine_queue_opt = ""

    run_opts.command = args.command

    return [args, run_opts]

# Called from TrainOneIteration, this model does one iteration of training
# with 'num_jobs' jobs, and
# writes files like exp/tdnn_a/24.{1,2,3,..<num_jobs>}.raw
def TrainNewModels(dir, iter, srand, num_jobs,
                   num_archives_processed, num_archives,
                   raw_model_string, egs_dir,
                   apply_deriv_weights,
                   left_deriv_truncate, right_deriv_truncate,
                   l2_regularize, xent_regularize, leaky_hmm_coefficient,
                   momentum, max_param_change,
                   shuffle_buffer_size, num_chunk_per_minibatch,
                   frame_subsampling_factor, truncate_deriv_weights,
                   cache_io_opts, run_opts):
    # We cannot easily use a single parallel SGE job to do the main training,
    # because the computation of which archive and which --frame option
    # to use for each job is a little complex, so we spawn each one separately.
    # this is no longer true for RNNs as we use do not use the --frame option
    # but we use the same script for consistency with FF-DNN code

    deriv_time_opts=""
    if left_deriv_truncate is not None:
        deriv_time_opts += " --optimization.min-deriv-time={0}".format(left_deriv_truncate)
    if right_deriv_truncate is not None:
        deriv_time_opts += " --optimization.max-deriv-time={0}".format(int(chunk-width-right_deriv_truncate))

    processes = []
    for job in range(1,num_jobs+1):
        k = num_archives_processed + job - 1 # k is a zero-based index that we will derive
                                               # the other indexes from.
        archive_index = (k % num_archives) + 1 # work out the 1-based archive index.
        frame_shift = (archive_index + k/num_archives) % frame_subsampling_factor
        # previous : frame_shift = (k/num_archives) % frame_subsampling_factor
        if job == 1:
            cur_cache_io_opts = cache_io_opts + " --write-cache={dir}/cache.{next_iter}".format(dir = dir, next_iter = iter + 1)
        else:
            cur_cache_io_opts = cache_io_opts

        process_handle = common_train_lib.RunKaldiCommand("""
{command} {train_queue_opt} {dir}/log/train.{iter}.{job}.log \
  nnet3-chain-train {parallel_train_opts} \
  --apply-deriv-weights={app_deriv_wts} \
  --l2-regularize={l2} --leaky-hmm-coefficient={leaky} \
  {cache_io_opts}  --xent-regularize={xent_reg} {deriv_time_opts} \
  --print-interval=10 --momentum={momentum} \
  --max-param-change={max_param_change} \
   "{raw_model}" {dir}/den.fst \
  "ark,bg:nnet3-chain-copy-egs --truncate-deriv-weights={trunc_deriv} --frame-shift={fr_shft} ark:{egs_dir}/cegs.{archive_index}.ark ark:- | nnet3-chain-shuffle-egs --buffer-size={shuffle_buffer_size} --srand={srand} ark:- ark:-| nnet3-chain-merge-egs --minibatch-size={num_chunk_per_minibatch} ark:- ark:- |" \
  {dir}/{next_iter}.{job}.raw
          """.format(command = run_opts.command,
                     train_queue_opt = run_opts.train_queue_opt,
                     dir = dir, iter = iter, srand = iter + srand, next_iter = iter + 1, job = job,
                     deriv_time_opts = deriv_time_opts,
                     trunc_deriv = truncate_deriv_weights,
                     app_deriv_wts = apply_deriv_weights,
                     fr_shft = frame_shift, l2 = l2_regularize,
                     xent_reg = xent_regularize, leaky = leaky_hmm_coefficient,
                     parallel_train_opts = run_opts.parallel_train_opts,
                     momentum = momentum, max_param_change = max_param_change,
                     raw_model = raw_model_string,
                     egs_dir = egs_dir, archive_index = archive_index,
                     shuffle_buffer_size = shuffle_buffer_size,
                     cache_io_opts = cur_cache_io_opts,
                     num_chunk_per_minibatch = num_chunk_per_minibatch),
          wait = False)

        processes.append(process_handle)

    all_success = True
    for process in processes:
        process.wait()
        [stdout_value, stderr_value] = process.communicate()
        if stderr_value.strip() != '':
            print(stderr_value)
        if process.returncode != 0:
            all_success = False

    if not all_success:
        open('{0}/.error'.format(dir), 'w').close()
        raise Exception("There was error during training iteration {0}".format(iter))

def TrainOneIteration(dir, iter, srand, egs_dir,
                      num_jobs, num_archives_processed, num_archives,
                      learning_rate, shrinkage_value, num_chunk_per_minibatch,
                      num_hidden_layers, add_layers_period,
                      apply_deriv_weights, left_deriv_truncate, right_deriv_truncate,
                      l2_regularize, xent_regularize, leaky_hmm_coefficient,
                      momentum, max_param_change, shuffle_buffer_size,
                      frame_subsampling_factor, truncate_deriv_weights,
                      run_opts):

    # Set off jobs doing some diagnostics, in the background.
    # Use the egs dir from the previous iteration for the diagnostics
    logger.info("Training neural net (pass {0})".format(iter))

    # check if different iterations use the same random seed
    if os.path.exists('{0}/srand'.format(dir)):
        try:
            saved_srand = int(open('{0}/srand'.format(dir), 'r').readline().strip())
        except IOError, ValueError:
            raise Exception('Exception while reading the random seed for training')
        if srand != saved_srand:
            logger.warning("The random seed provided to this iteration (srand={0}) is different from the one saved last time (srand={1}). Using srand={0}.".format(srand, saved_srand))
    else:
        f = open('{0}/srand'.format(dir), 'w')
        f.write(str(srand))
        f.close()

    chain_lib.ComputeTrainCvProbabilities(dir, iter, egs_dir,
            l2_regularize, xent_regularize, leaky_hmm_coefficient, run_opts)

    if iter > 0:
        chain_lib.ComputeProgress(dir, iter, run_opts)

    if iter > 0 and (iter <= (num_hidden_layers-1) * add_layers_period) and (iter % add_layers_period == 0):

        do_average = False # if we've just mixed up, don't do averaging but take the
                           # best.
        cur_num_hidden_layers = 1 + iter / add_layers_period
        config_file = "{0}/configs/layer{1}.config".format(dir, cur_num_hidden_layers)
        raw_model_string = "nnet3-am-copy --raw=true --learning-rate={lr} {dir}/{iter}.mdl - | nnet3-init --srand={srand} - {config} - |".format(lr=learning_rate, dir=dir, iter=iter, srand=iter + srand, config=config_file)
        cache_io_opts = ""
    else:
        do_average = True
        if iter == 0:
            do_average = False   # on iteration 0, pick the best, don't average.
        raw_model_string = "nnet3-am-copy --raw=true --learning-rate={0} {1}/{2}.mdl - |".format(learning_rate, dir, iter)
        cache_io_opts = "--read-cache={dir}/cache.{iter}".format(dir = dir, iter = iter)

    if do_average:
        cur_num_chunk_per_minibatch = num_chunk_per_minibatch
        cur_max_param_change = max_param_change
    else:
        # on iteration zero or when we just added a layer, use a smaller minibatch
        # size (and we will later choose the output of just one of the jobs): the
        # model-averaging isn't always helpful when the model is changing too fast
        # (i.e. it can worsen the objective function), and the smaller minibatch
        # size will help to keep the update stable.
        cur_num_chunk_per_minibatch = num_chunk_per_minibatch / 2
        cur_max_param_change = float(max_param_change) / math.sqrt(2)

    TrainNewModels(dir, iter, srand, num_jobs, num_archives_processed, num_archives,
                   raw_model_string, egs_dir,
                   apply_deriv_weights,
                   left_deriv_truncate, right_deriv_truncate,
                   l2_regularize, xent_regularize, leaky_hmm_coefficient,
                   momentum, cur_max_param_change,
                   shuffle_buffer_size, cur_num_chunk_per_minibatch,
                   frame_subsampling_factor, truncate_deriv_weights,
                   cache_io_opts, run_opts)

    [models_to_average, best_model] = common_train_lib.GetSuccessfulModels(num_jobs, '{0}/log/train.{1}.%.log'.format(dir,iter))
    nnets_list = []
    for n in models_to_average:
        nnets_list.append("{0}/{1}.{2}.raw".format(dir, iter + 1, n))

    if do_average:
        # average the output of the different jobs.
        common_train_lib.RunKaldiCommand("""
{command} {dir}/log/average.{iter}.log \
nnet3-average {nnet_list} - \| \
nnet3-am-copy --scale={shrink} --set-raw-nnet=- {dir}/{iter}.mdl {dir}/{new_iter}.mdl
        """.format(command = run_opts.command,
                   dir = dir,
                   iter = iter,
                   nnet_list = " ".join(nnets_list),
                   shrink = shrinkage_value,
                   new_iter = iter + 1))

    else:
        # choose the best model from different jobs
        common_train_lib.RunKaldiCommand("""
{command} {dir}/log/select.{iter}.log \
    nnet3-am-copy --scale={shrink} --set-raw-nnet={dir}/{next_iter}.{best_model_index}.raw  {dir}/{iter}.mdl {dir}/{next_iter}.mdl
        """.format(command = run_opts.command,
                   dir = dir, iter = iter, next_iter = iter + 1,
                   shrink = shrinkage_value, best_model_index =  best_model))

    try:
        for i in range(1, num_jobs + 1):
            os.remove("{0}/{1}.{2}.raw".format(dir, iter + 1, i))
    except OSError:
        raise Exception("Error while trying to delete the raw models")

    new_model = "{0}/{1}.mdl".format(dir, iter + 1)

    if not os.path.isfile(new_model):
        raise Exception("Could not find {0}, at the end of iteration {1}".format(new_model, iter))
    elif os.stat(new_model).st_size == 0:
        raise Exception("{0} has size 0. Something went wrong in iteration {1}".format(new_model, iter))
    if os.path.exists("{0}/cache.{1}".format(dir, iter)):
        os.remove("{0}/cache.{1}".format(dir, iter))

def CheckForRequiredFiles(feat_dir, tree_dir, lat_dir):
    for file in ['{0}/feats.scp'.format(feat_dir), '{0}/ali.1.gz'.format(tree_dir),
                 '{0}/final.mdl'.format(tree_dir), '{0}/tree'.format(tree_dir),
                 '{0}/lat.1.gz'.format(lat_dir), '{0}/final.mdl'.format(lat_dir),
                 '{0}/num_jobs'.format(lat_dir), '{0}/splice_opts'.format(lat_dir)]:
        if not os.path.isfile(file):
            raise Exception('Expected {0} to exist.'.format(file))


# args is a Namespace with the required parameters
def Train(args, run_opts):
    arg_string = pprint.pformat(vars(args))
    logger.info("Arguments for the experiment\n{0}".format(arg_string))

    # Check files
    CheckForRequiredFiles(args.feat_dir, args.tree_dir, args.lat_dir)

    # Set some variables.
    num_jobs = common_train_lib.GetNumberOfJobs(args.tree_dir)
    feat_dim = common_train_lib.GetFeatDim(args.feat_dir)
    ivector_dim = common_train_lib.GetIvectorDim(args.online_ivector_dir)

    # split the training data into parts for individual jobs
    # we will use the same number of jobs as that used for alignment
    common_train_lib.SplitData(args.feat_dir, num_jobs)
    shutil.copy('{0}/tree'.format(args.tree_dir), args.dir)
    f = open('{0}/num_jobs'.format(args.dir), 'w')
    f.write(str(num_jobs))
    f.close()

    config_dir = '{0}/configs'.format(args.dir)
    var_file = '{0}/vars'.format(config_dir)

    variables = common_train_lib.ParseGenericConfigVarsFile(var_file)

    # Set some variables.

    try:
        model_left_context = variables['model_left_context']
        model_right_context = variables['model_right_context']
        num_hidden_layers = variables['num_hidden_layers']
    except KeyError as e:
        raise Exception("KeyError {0}: Variables need to be defined in {1}".format(
            str(e), '{0}/configs'.format(args.dir)))

    left_context = args.chunk_left_context + model_left_context
    right_context = args.chunk_right_context + model_right_context

    # Initialize as "raw" nnet, prior to training the LDA-like preconditioning
    # matrix.  This first config just does any initial splicing that we do;
    # we do this as it's a convenient way to get the stats for the 'lda-like'
    # transform.
    if (args.stage <= -6):
        logger.info("Creating phone language-model")
        chain_lib.CreatePhoneLm(args.dir, args.tree_dir, run_opts, lm_opts = args.lm_opts)

    if (args.stage <= -5):
        logger.info("Creating denominator FST")
        chain_lib.CreateDenominatorFst(args.dir, args.tree_dir, run_opts)

    if (args.stage <= -4):
        logger.info("Initializing a basic network for estimating preconditioning matrix")
        common_train_lib.RunKaldiCommand("""
{command} {dir}/log/nnet_init.log \
    nnet3-init --srand=-2 {dir}/configs/init.config {dir}/init.raw
    """.format(command = run_opts.command,
               dir = args.dir))

    left_context = args.chunk_left_context + model_left_context
    right_context = args.chunk_right_context + model_right_context
    egs_left_context = left_context + args.frame_subsampling_factor/2
    egs_right_context = right_context + args.frame_subsampling_factor/2

    default_egs_dir = '{0}/egs'.format(args.dir)
    if (args.stage <= -3) and args.egs_dir is None:
        logger.info("Generating egs")
        # this is where get_egs.sh is called.
        chain_lib.GenerateChainEgs(args.dir, args.feat_dir, args.lat_dir, default_egs_dir,
                                   egs_left_context, egs_right_context,
                                   run_opts,
                                   left_tolerance = args.left_tolerance,
                                   right_tolerance = args.right_tolerance,
                                   frame_subsampling_factor = args.frame_subsampling_factor,
                                   alignment_subsampling_factor = args.alignment_subsampling_factor,
                                   frames_per_eg = args.chunk_width,
                                   egs_opts = args.egs_opts,
                                   cmvn_opts = args.cmvn_opts,
                                   online_ivector_dir = args.online_ivector_dir,
                                   frames_per_iter = args.frames_per_iter,
                                   srand = args.srand,
                                   transform_dir = args.transform_dir,
                                   stage = args.egs_stage)

    if args.egs_dir is None:
        egs_dir = default_egs_dir
    else:
        egs_dir = args.egs_dir

    [egs_left_context, egs_right_context,
     frames_per_eg, num_archives] = (
             common_train_lib.VerifyEgsDir(egs_dir, feat_dim, ivector_dim,
                                           egs_left_context, egs_right_context) )
    assert(args.chunk_width == frames_per_eg)
    num_archives_expanded = num_archives * args.frame_subsampling_factor

    if (args.num_jobs_final > num_archives_expanded):
        raise Exception('num_jobs_final cannot exceed the expanded number of archives')

    # copy the properties of the egs to dir for
    # use during decoding
    common_train_lib.CopyEgsPropertiesToExpDir(egs_dir, args.dir)

    if (args.stage <= -2):
        logger.info('Computing the preconditioning matrix for input features')

        chain_lib.ComputePreconditioningMatrix(args.dir, egs_dir, num_archives, run_opts,
                                               max_lda_jobs = args.max_lda_jobs,
                                               rand_prune = args.rand_prune)

    if (args.stage <= -1):
        logger.info("Preparing the initial acoustic model.")
        chain_lib.PrepareInitialAcousticModel(args.dir, run_opts)

    file_handle = open("{0}/frame_subsampling_factor".format(args.dir),"w")
    file_handle.write(str(args.frame_subsampling_factor))
    file_handle.close()

    # set num_iters so that as close as possible, we process the data $num_epochs
    # times, i.e. $num_iters*$avg_num_jobs) == $num_epochs*$num_archives,
    # where avg_num_jobs=(num_jobs_initial+num_jobs_final)/2.
    num_archives_to_process = args.num_epochs * num_archives_expanded
    num_archives_processed = 0
    num_iters=(num_archives_to_process * 2) / (args.num_jobs_initial + args.num_jobs_final)

    num_iters_combine = common_train_lib.VerifyIterations(
                                         num_iters, args.num_epochs,
                                         num_hidden_layers, num_archives_expanded,
                                         args.max_models_combine, args.add_layers_period,
                                         args.num_jobs_final)

    learning_rate = (lambda iter, current_num_jobs, num_archives_processed:
                        common_train_lib.GetLearningRate(
                                         iter, current_num_jobs, num_iters,
                                         num_archives_processed,
                                         num_archives_to_process,
                                         args.initial_effective_lrate,
                                         args.final_effective_lrate)
                    )

    min_deriv_time = None
    max_deriv_time = None
    if not args.deriv_truncate_margin is None:
        min_deriv_time = -args.deriv_truncate_margin
        max_deriv_time = args.chunk_width - 1 + args.deriv_truncate_margin

    logger.info("Training will run for {0} epochs = {1} iterations".format(args.num_epochs, num_iters))
    for iter in range(num_iters):
        if (args.exit_stage is not None) and (iter == args.exit_stage):
            logger.info("Exiting early due to --exit-stage {0}".format(iter))
            return
        current_num_jobs = int(0.5 + args.num_jobs_initial + (args.num_jobs_final - args.num_jobs_initial) * float(iter) / num_iters)

        if args.stage <= iter:
            model_file = "{dir}/{iter}.mdl".format(dir = args.dir, iter = iter)
            shrinkage_value = (args.shrink_value
                               if common_train_lib.DoShrinkage(iter, model_file,
                                                               args.shrink_nonlinearity,
                                                               args.shrink_threshold)
                               else 1
                               )
            logger.info("On iteration {0}, learning rate is {1} and shrink value is {2}.".format(iter, learning_rate(iter, current_num_jobs, num_archives_processed), shrinkage_value))

            TrainOneIteration(dir = args.dir,
                              iter = iter,
                              srand = args.srand,
                              egs_dir = egs_dir,
                              num_jobs = current_num_jobs,
                              num_archives_processed = num_archives_processed,
                              num_archives = num_archives,
                              learning_rate = learning_rate(iter, current_num_jobs, num_archives_processed),
                              shrinkage_value = shrinkage_value,
                              num_chunk_per_minibatch = args.num_chunk_per_minibatch,
                              num_hidden_layers = num_hidden_layers,
                              add_layers_period = args.add_layers_period,
                              left_context = left_context,
                              right_context = right_context,
                              apply_deriv_weights = args.apply_deriv_weights,
                              min_deriv_time = min_deriv_time,
                              max_deriv_time = max_deriv_time,
                              l2_regularize = args.l2_regularize,
                              xent_regularize = args.xent_regularize,
                              leaky_hmm_coefficient = args.leaky_hmm_coefficient,
                              momentum = args.momentum,
                              max_param_change = args.max_param_change,
                              shuffle_buffer_size = args.shuffle_buffer_size,
                              frame_subsampling_factor = args.frame_subsampling_factor,
                              truncate_deriv_weights = args.truncate_deriv_weights,
                              run_opts = run_opts)
            if args.cleanup:
                # do a clean up everythin but the last 2 models, under certain conditions
                common_train_lib.RemoveModel(
                                 args.dir, iter-2, num_iters, num_iters_combine,
                                 args.preserve_model_interval)

            if args.email is not None:
                reporting_iter_interval = num_iters * args.reporting_interval
                if iter % reporting_iter_interval == 0:
                # lets do some reporting
                    [report, times, data] = nnet3_log_parse.GenerateAccuracyReport(args.dir, key="log-probability")
                    message = report
                    subject = "Update : Expt {dir} : Iter {iter}".format(dir = args.dir, iter = iter)
                    common_train_lib.SendMail(message, subject, args.email)

        num_archives_processed = num_archives_processed + current_num_jobs

    if args.stage <= num_iters:
        logger.info("Doing final combination to produce final.mdl")
        chain_lib.CombineModels(dir = args.dir,
                                num_iters = num_iters,
                                num_iters_combine = num_iters_combine,
                                num_chunk_per_minibatch = args.num_chunk_per_minibatch,
                                egs_dir = egs_dir,
                                left_context = left_context,
                                right_context = right_context,
                                leaky_hmm_coefficient = args.leaky_hmm_coefficient,
                                l2_regularize = args.l2_regularize,
                                xent_regularize = args.xent_regularize,
                                run_opts = run_opts)

    if args.cleanup:
        logger.info("Cleaning up the experiment directory {0}".format(args.dir))
        remove_egs = args.remove_egs
        if args.egs_dir is not None:
            # this egs_dir was not created by this experiment so we will not
            # delete it
            remove_egs = False

        common_train_lib.CleanNnetDir(args.dir, num_iters, egs_dir,
                         preserve_model_interval = args.preserve_model_interval,
                         remove_egs = remove_egs)

    # do some reporting
    [report, times, data] = nnet3_log_parse.GenerateAccuracyReport(args.dir, "log-probability")
    if args.email is not None:
        common_train_lib.SendMail(report, "Update : Expt {0} : complete".format(args.dir), args.email)

    report_handle = open("{dir}/accuracy.report".format(dir = args.dir), "w")
    report_handle.write(report)
    report_handle.close()

    os.system("steps/info/chain_dir_info.pl " + args.dir)

def Main():
    [args, run_opts] = GetArgs()
    try:
        Train(args, run_opts)
    except Exception as e:
        if args.email is not None:
            message = "Training session for experiment {dir} died due to an error.".format(dir = args.dir)
            common_train_lib.SendMail(message, message, args.email)
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    Main()
