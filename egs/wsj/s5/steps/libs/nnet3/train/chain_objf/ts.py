

# Copyright 2016    Vijayaditya Peddinti.
#           2016    Vimal Manohar
# Apache 2.0.

""" This is a module with methods which will be used by scripts for
teacher-student training of deep neural network acoustic model with
sequence KL objective.
"""

import logging
import math
import os
import sys

import libs.common as common_lib
import libs.nnet3.train.common as common_train_lib
from . import acoustic_model as chain_lib

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def generate_chain_egs(dir, data, lat_dir, egs_dir,
                       left_context, right_context,
                       run_opts, stage=0,
                       left_context_initial=-1, right_context_final=-1,
                       frame_subsampling_factor=3,
                       online_ivector_dir=None,
                       frames_per_iter=20000, frames_per_eg_str="20", srand=0,
                       egs_opts=None, cmvn_opts=None, transform_dir=None):
    """Wrapper for steps/nnet3/chain/get_egs_ts.sh

    See options in that script.
    """

    common_lib.execute_command(
        """steps/nnet3/chain/get_egs_ts.sh {egs_opts} \
                --cmd "{command}" \
                --cmvn-opts "{cmvn_opts}" \
                --transform-dir "{transform_dir}" \
                --online-ivector-dir "{ivector_dir}" \
                --left-context {left_context} \
                --right-context {right_context} \
                --left-context-initial {left_context_initial} \
                --right-context-final {right_context_final} \
                --frame-subsampling-factor {frame_subsampling_factor} \
                --stage {stage} \
                --frames-per-iter {frames_per_iter} \
                --frames-per-eg {frames_per_eg_str} \
                --srand {srand} \
                {data} {dir} {lat_dir} {egs_dir}""".format(
                    command=run_opts.egs_command,
                    cmvn_opts=cmvn_opts if cmvn_opts is not None else '',
                    transform_dir=(transform_dir
                                   if transform_dir is not None
                                   else ''),
                    ivector_dir=(online_ivector_dir
                                 if online_ivector_dir is not None
                                 else ''),
                    left_context=left_context,
                    right_context=right_context,
                    left_context_initial=left_context_initial,
                    right_context_final=right_context_final,
                    frame_subsampling_factor=frame_subsampling_factor,
                    stage=stage, frames_per_iter=frames_per_iter,
                    frames_per_eg_str=frames_per_eg_str, srand=srand,
                    data=data, lat_dir=lat_dir, dir=dir, egs_dir=egs_dir,
                    egs_opts=egs_opts if egs_opts is not None else ''))


#def train_new_models(dir, iter, srand, num_jobs,
#                     num_archives_processed, num_archives,
#                     raw_model_string, egs_dir,
#                     apply_deriv_weights,
#                     min_deriv_time, max_deriv_time_relative,
#                     l2_regularize, xent_regularize, leaky_hmm_coefficient,
#                     momentum, max_param_change,
#                     shuffle_buffer_size, num_chunk_per_minibatch_str,
#                     frame_subsampling_factor, truncate_deriv_weights, run_opts,
#                     backstitch_training_scale=0.0, backstitch_training_interval=1,
#                     use_multitask_egs=False, objective_opts=""):
#    """
#    Called from train_one_iteration(), this method trains new models
#    with 'num_jobs' jobs, and
#    writes files like exp/tdnn_a/24.{1,2,3,..<num_jobs>}.raw
#
#    We cannot easily use a single parallel SGE job to do the main training,
#    because the computation of which archive and which --frame option
#    to use for each job is a little complex, so we spawn each one separately.
#    this is no longer true for RNNs as we use do not use the --frame option
#    but we use the same script for consistency with FF-DNN code
#
#    use_multitask_egs : True, if different examples used to train multiple
#                        tasks or outputs, e.g.multilingual training.
#                        multilingual egs can be generated using get_egs.sh and
#                        steps/nnet3/multilingual/allocate_multilingual_examples.py,
#                        those are the top-level scripts.
#    """
#
#    deriv_time_opts = []
#    if min_deriv_time is not None:
#        deriv_time_opts.append("--optimization.min-deriv-time={0}".format(
#                                    min_deriv_time))
#    if max_deriv_time_relative is not None:
#        deriv_time_opts.append("--optimization.max-deriv-time-relative={0}".format(
#                                    int(max_deriv_time_relative)))
#
#    threads = []
#    # the GPU timing info is only printed if we use the --verbose=1 flag; this
#    # slows down the computation slightly, so don't accumulate it on every
#    # iteration.  Don't do it on iteration 0 either, because we use a smaller
#    # than normal minibatch size, and people may get confused thinking it's
#    # slower for iteration 0 because of the verbose option.
#    verbose_opt = ("--verbose=1" if iter % 20 == 0 and iter > 0 else "")
#
#    for job in range(1, num_jobs+1):
#        # k is a zero-based index that we will derive the other indexes from.
#        k = num_archives_processed + job - 1
#        # work out the 1-based archive index.
#        archive_index = (k % num_archives) + 1
#        # previous : frame_shift = (k/num_archives) % frame_subsampling_factor
#        frame_shift = ((archive_index + k/num_archives)
#                       % frame_subsampling_factor)
#
#        multitask_egs_opts = common_train_lib.get_multitask_egs_opts(
#            egs_dir,
#            egs_prefix="cegs.",
#            archive_index=archive_index,
#            use_multitask_egs=use_multitask_egs)
#        scp_or_ark = "scp" if use_multitask_egs else "ark"
#        cache_io_opts = (("--read-cache={dir}/cache.{iter}".format(dir=dir,
#                                                                  iter=iter)
#                          if iter > 0 else "") +
#                         (" --write-cache={0}/cache.{1}".format(dir, iter + 1)
#                          if job == 1 else ""))
#
#        if truncate_deriv_weights > 0:
#            raise NotImplementedError
#
#        thread = common_lib.background_command(
#            """{command} {train_queue_opt} {dir}/log/train.{iter}.{job}.log \
#                    nnet3-chain-train-post {parallel_train_opts} {verbose_opt} \
#                    --apply-deriv-weights={app_deriv_wts} {objective_opts} \
#                    --l2-regularize={l2} --leaky-hmm-coefficient={leaky} \
#                    {cache_io_opts}  --xent-regularize={xent_reg} \
#                    {deriv_time_opts} \
#                    --print-interval=10 --momentum={momentum} \
#                    --max-param-change={max_param_change} \
#                    --backstitch-training-scale={backstitch_training_scale} \
#                    --backstitch-training-interval={backstitch_training_interval} \
#                    --l2-regularize-factor={l2_regularize_factor} \
#                    --srand={srand} \
#                    "{raw_model}" {dir}/den.fst \
#                    "ark,bg:nnet3-copy-egs {multitask_egs_opts} \
#                        --frame-shift={fr_shft} \
#                        {scp_or_ark}:{egs_dir}/egs.{archive_index}.{scp_or_ark} ark:- | \
#                        nnet3-shuffle-egs --buffer-size={buf_size} \
#                        --srand={srand} ark:- ark:- | nnet3-merge-egs --sort-by-t \
#                        --minibatch-size={num_chunk_per_mb} ark:- ark:- |" \
#                    {dir}/{next_iter}.{job}.raw""".format(
#                        command=run_opts.command,
#                        train_queue_opt=run_opts.train_queue_opt,
#                        dir=dir, iter=iter, srand=iter + srand,
#                        next_iter=iter + 1, job=job,
#                        deriv_time_opts=" ".join(deriv_time_opts),
#                        app_deriv_wts=apply_deriv_weights,
#                        fr_shft=frame_shift, l2=l2_regularize,
#                        xent_reg=xent_regularize, leaky=leaky_hmm_coefficient,
#                        cache_io_opts=cache_io_opts,
#                        parallel_train_opts=run_opts.parallel_train_opts,
#                        verbose_opt=verbose_opt,
#                        momentum=momentum, max_param_change=max_param_change,
#                        backstitch_training_scale=backstitch_training_scale,
#                        backstitch_training_interval=backstitch_training_interval,
#                        l2_regularize_factor=1.0/num_jobs,
#                        raw_model=raw_model_string,
#                        egs_dir=egs_dir, archive_index=archive_index,
#                        buf_size=shuffle_buffer_size,
#                        num_chunk_per_mb=num_chunk_per_minibatch_str,
#                        multitask_egs_opts=multitask_egs_opts,
#                        scp_or_ark=scp_or_ark,
#                        objective_opts=objective_opts),
#            require_zero_status=True)
#
#        threads.append(thread)
#
#    for thread in threads:
#        thread.join()
#
#
#def train_one_iteration(dir, iter, srand, egs_dir,
#                        num_jobs, num_archives_processed, num_archives,
#                        learning_rate, shrinkage_value,
#                        num_chunk_per_minibatch_str,
#                        apply_deriv_weights, min_deriv_time,
#                        max_deriv_time_relative,
#                        l2_regularize, xent_regularize,
#                        leaky_hmm_coefficient,
#                        momentum, max_param_change, shuffle_buffer_size,
#                        frame_subsampling_factor, truncate_deriv_weights,
#                        run_opts, dropout_edit_string="",
#                        backstitch_training_scale=0.0, backstitch_training_interval=1,
#                        use_multitask_egs=False,
#                        objective_opts=""):
#    """ Called from steps/nnet3/chain/train_ts.py for one iteration for
#    neural network training with LF-MMI objective
#
#    """
#
#    # Set off jobs doing some diagnostics, in the background.
#    # Use the egs dir from the previous iteration for the diagnostics
#    # check if different iterations use the same random seed
#    if os.path.exists('{0}/srand'.format(dir)):
#        try:
#            saved_srand = int(open('{0}/srand'.format(dir)).readline().strip())
#        except (IOError, ValueError):
#            logger.error("Exception while reading the random seed "
#                         "for training")
#            raise
#        if srand != saved_srand:
#            logger.warning("The random seed provided to this iteration "
#                           "(srand={0}) is different from the one saved last "
#                           "time (srand={1}). Using srand={0}.".format(
#                               srand, saved_srand))
#    else:
#        with open('{0}/srand'.format(dir), 'w') as f:
#            f.write(str(srand))
#
#    # Sets off some background jobs to compute train and
#    # validation set objectives
#    compute_train_cv_probabilities(
#        dir=dir, iter=iter, egs_dir=egs_dir,
#        l2_regularize=l2_regularize, xent_regularize=xent_regularize,
#        leaky_hmm_coefficient=leaky_hmm_coefficient, run_opts=run_opts,
#        use_multitask_egs=use_multitask_egs,
#        objective_opts=objective_opts)
#
#    if iter > 0:
#        # Runs in the background
#        chain_lib.compute_progress(dir, iter, run_opts)
#
#    do_average = (iter > 0)
#
#    raw_model_string = ("nnet3-am-copy --raw=true --learning-rate={0} "
#                        "--scale={1} {2}/{3}.mdl - |".format(
#                            learning_rate, shrinkage_value, dir, iter))
#
#    if do_average:
#        cur_num_chunk_per_minibatch_str = num_chunk_per_minibatch_str
#        cur_max_param_change = max_param_change
#    else:
#        # on iteration zero, use a smaller minibatch size (and we will later
#        # choose the output of just one of the jobs): the model-averaging isn't
#        # always helpful when the model is changing too fast (i.e. it can worsen
#        # the objective function), and the smaller minibatch size will help to
#        # keep the update stable.
#        cur_num_chunk_per_minibatch_str = common_train_lib.halve_minibatch_size_str(
#            num_chunk_per_minibatch_str)
#        cur_max_param_change = float(max_param_change) / math.sqrt(2)
#
#    raw_model_string = raw_model_string + dropout_edit_string
#
#    shrink_info_str = ''
#    if shrinkage_value != 1.0:
#        shrink_info_str = ' and shrink value is {0}'.format(shrinkage_value)
#
#    objf_info = "" if objective_opts == "" else (
#        "and objective_opts=" + objective_opts)
#    logger.info("On iteration {0}, learning rate is {1}"
#                "{shrink_info} {objf_info}.".format(
#                    iter, learning_rate,
#                    shrink_info=shrink_info_str, objf_info=objf_info))
#
#    train_new_models(dir=dir, iter=iter, srand=srand, num_jobs=num_jobs,
#                     num_archives_processed=num_archives_processed,
#                     num_archives=num_archives,
#                     raw_model_string=raw_model_string,
#                     egs_dir=egs_dir,
#                     apply_deriv_weights=apply_deriv_weights,
#                     min_deriv_time=min_deriv_time,
#                     max_deriv_time_relative=max_deriv_time_relative,
#                     l2_regularize=l2_regularize,
#                     xent_regularize=xent_regularize,
#                     leaky_hmm_coefficient=leaky_hmm_coefficient,
#                     momentum=momentum,
#                     max_param_change=cur_max_param_change,
#                     shuffle_buffer_size=shuffle_buffer_size,
#                     num_chunk_per_minibatch_str=cur_num_chunk_per_minibatch_str,
#                     frame_subsampling_factor=frame_subsampling_factor,
#                     truncate_deriv_weights=truncate_deriv_weights,
#                     run_opts=run_opts,
#                     # linearly increase backstitch_training_scale during the
#                     # first few iterations (hard-coded as 15)
#                     backstitch_training_scale=(backstitch_training_scale *
#                         iter / 15 if iter < 15 else backstitch_training_scale),
#                     backstitch_training_interval=backstitch_training_interval,
#                     use_multitask_egs=use_multitask_egs,
#                     objective_opts=objective_opts)
#
#    [models_to_average, best_model] = common_train_lib.get_successful_models(
#         num_jobs, '{0}/log/train.{1}.%.log'.format(dir, iter))
#    nnets_list = []
#    for n in models_to_average:
#        nnets_list.append("{0}/{1}.{2}.raw".format(dir, iter + 1, n))
#
#    if do_average:
#        # average the output of the different jobs.
#        common_train_lib.get_average_nnet_model(
#            dir=dir, iter=iter,
#            nnets_list=" ".join(nnets_list),
#            run_opts=run_opts)
#
#    else:
#        # choose the best model from different jobs
#        common_train_lib.get_best_nnet_model(
#            dir=dir, iter=iter,
#            best_model_index=best_model,
#            run_opts=run_opts)
#
#    try:
#        for i in range(1, num_jobs + 1):
#            os.remove("{0}/{1}.{2}.raw".format(dir, iter + 1, i))
#    except OSError:
#        raise Exception("Error while trying to delete the raw models")
#
#    new_model = "{0}/{1}.mdl".format(dir, iter + 1)
#
#    if not os.path.isfile(new_model):
#        raise Exception("Could not find {0}, at the end of "
#                        "iteration {1}".format(new_model, iter))
#    elif os.stat(new_model).st_size == 0:
#        raise Exception("{0} has size 0. Something went wrong in "
#                        "iteration {1}".format(new_model, iter))
#    if os.path.exists("{0}/cache.{1}".format(dir, iter)):
#        os.remove("{0}/cache.{1}".format(dir, iter))
#
#
#def compute_train_cv_probabilities(dir, iter, egs_dir, l2_regularize,
#                                   xent_regularize, leaky_hmm_coefficient,
#                                   run_opts,
#                                   use_multitask_egs=False,
#                                   objective_opts=""):
#    model = '{0}/{1}.mdl'.format(dir, iter)
#    scp_or_ark = "scp" if use_multitask_egs else "ark"
#    egs_suffix = ".scp" if use_multitask_egs else ".egs"
#
#    multitask_egs_opts = common_train_lib.get_multitask_egs_opts(
#                             egs_dir,
#                             egs_prefix="valid_diagnostic.",
#                             use_multitask_egs=use_multitask_egs)
#
#    import re
#    objective_opts = re.sub(r"--mmi-factor=0.0 ", "--mmi-factor=1e-10 ",
#                            objective_opts)
#
#    common_lib.background_command(
#        """{command} {dir}/log/compute_prob_valid.{iter}.log \
#                nnet3-chain-compute-prob-post --l2-regularize={l2} {objective_opts} \
#                --leaky-hmm-coefficient={leaky} --xent-regularize={xent_reg} \
#                {model} {dir}/den.fst \
#                "ark,bg:nnet3-copy-egs {multitask_egs_opts} {scp_or_ark}:{egs_dir}/valid_diagnostic{egs_suffix} \
#                    ark:- | nnet3-merge-egs --sort-by-t --minibatch-size=1:64 ark:- ark:- |" \
#        """.format(command=run_opts.command, dir=dir, iter=iter, model=model,
#                   l2=l2_regularize, leaky=leaky_hmm_coefficient,
#                   xent_reg=xent_regularize,
#                   egs_dir=egs_dir,
#                   multitask_egs_opts=multitask_egs_opts,
#                   scp_or_ark=scp_or_ark, egs_suffix=egs_suffix,
#                   objective_opts=objective_opts))
#
#    multitask_egs_opts = common_train_lib.get_multitask_egs_opts(
#                             egs_dir,
#                             egs_prefix="train_diagnostic.",
#                             use_multitask_egs=use_multitask_egs)
#
#    common_lib.background_command(
#        """{command} {dir}/log/compute_prob_train.{iter}.log \
#                nnet3-chain-compute-prob-post --l2-regularize={l2} {objective_opts} \
#                --leaky-hmm-coefficient={leaky} --xent-regularize={xent_reg} \
#                "nnet3-am-copy --raw=true {model} - |" {dir}/den.fst \
#                "ark,bg:nnet3-copy-egs {multitask_egs_opts} {scp_or_ark}:{egs_dir}/train_diagnostic{egs_suffix} \
#                    ark:- | nnet3-merge-egs --sort-by-t --minibatch-size=1:64 ark:- ark:- |" \
#        """.format(command=run_opts.command, dir=dir, iter=iter, model=model,
#                   l2=l2_regularize, leaky=leaky_hmm_coefficient,
#                   xent_reg=xent_regularize,
#                   egs_dir=egs_dir,
#                   multitask_egs_opts=multitask_egs_opts,
#                   scp_or_ark=scp_or_ark, egs_suffix=egs_suffix,
#                   objective_opts=objective_opts))
#
#
#def combine_models(dir, num_iters, models_to_combine, num_chunk_per_minibatch_str,
#                   egs_dir, leaky_hmm_coefficient, l2_regularize,
#                   xent_regularize, run_opts,
#                   max_objective_evaluations=30,
#                   use_multitask_egs=False,
#                   objective_opts=""):
#    """ Function to do model combination
#
#    In the nnet3 setup, the logic
#    for doing averaging of subsets of the models in the case where
#    there are too many models to reliably esetimate interpolation
#    factors (max_models_combine) is moved into the nnet3-combine.
#    """
#    raw_model_strings = []
#    logger.info("Combining {0} models.".format(models_to_combine))
#
#    models_to_combine.add(num_iters)
#
#    for iter in sorted(models_to_combine):
#        model_file = '{0}/{1}.mdl'.format(dir, iter)
#        if os.path.exists(model_file):
#            # we used to copy them with nnet3-am-copy --raw=true, but now
#            # the raw-model-reading code discards the other stuff itself.
#            raw_model_strings.append(model_file)
#        else:
#            print("{0}: warning: model file {1} does not exist "
#                  "(final combination)".format(sys.argv[0], model_file))
#
#    scp_or_ark = "scp" if use_multitask_egs else "ark"
#    egs_suffix = ".scp" if use_multitask_egs else ".egs"
#
#    multitask_egs_opts = common_train_lib.get_multitask_egs_opts(
#                             egs_dir,
#                             egs_prefix="combine.",
#                             use_multitask_egs=use_multitask_egs)
#
#    # We reverse the order of the raw model strings so that the freshest one
#    # goes first.  This is important for systems that include batch
#    # normalization-- it means that the freshest batch-norm stats are used.
#    # Since the batch-norm stats are not technically parameters, they are not
#    # combined in the combination code, they are just obtained from the first
#    # model.
#    raw_model_strings = list(reversed(raw_model_strings))
#
#    common_lib.execute_command(
#        """{command} {combine_queue_opt} {dir}/log/combine.log \
#                nnet3-chain-combine-post {objective_opts} \
#                --max-objective-evaluations={max_objective_evaluations} \
#                --l2-regularize={l2} --leaky-hmm-coefficient={leaky} \
#                --verbose=3 {dir}/den.fst {raw_models} \
#                "ark,bg:nnet3-copy-egs {multitask_egs_opts} {scp_or_ark}:{egs_dir}/combine{egs_suffix} ark:- | \
#                    nnet3-merge-egs --sort-by-t --minibatch-size={num_chunk_per_mb} \
#                    ark:- ark:- |" - \| \
#                nnet3-am-copy --set-raw-nnet=- {dir}/{num_iters}.mdl \
#                {dir}/final.mdl""".format(
#                    command=run_opts.command,
#                    combine_queue_opt=run_opts.combine_queue_opt,
#                    max_objective_evaluations=max_objective_evaluations,
#                    l2=l2_regularize, leaky=leaky_hmm_coefficient,
#                    dir=dir, raw_models=" ".join(raw_model_strings),
#                    num_chunk_per_mb=num_chunk_per_minibatch_str,
#                    num_iters=num_iters,
#                    egs_dir=egs_dir,
#                    multitask_egs_opts=multitask_egs_opts,
#                    scp_or_ark=scp_or_ark, egs_suffix=egs_suffix,
#                    objective_opts=objective_opts))
#
#    # Compute the probability of the final, combined model with
#    # the same subset we used for the previous compute_probs, as the
#    # different subsets will lead to different probs.
#    compute_train_cv_probabilities(
#        dir=dir, iter='final', egs_dir=egs_dir,
#        l2_regularize=l2_regularize, xent_regularize=xent_regularize,
#        leaky_hmm_coefficient=leaky_hmm_coefficient,
#        run_opts=run_opts,
#        use_multitask_egs=use_multitask_egs,
#        objective_opts=objective_opts)

