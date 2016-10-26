#!/usr/bin/env python


# Copyright 2016 Vijayaditya Peddinti
#           2016 Vimal Manohar
# Apache 2.0.


import os
import subprocess
import argparse
import sys
import pprint
import logging
import imp
import traceback
import shlex
import random
import math
import glob

imp.load_source('data_lib', 'utils/data/data_lib.py')
imp.load_source('nnet3_log_parse', 'steps/nnet3/report/nnet3_log_parse_lib.py')
imp.load_source('train_lib', 'steps/nnet3/nnet3_train_lib.py')

import data_lib
import nnet3_log_parse
import train_lib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Getting egs for training')


def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="""
Generates training examples used to train the 'nnet3' network (and also the"""
" validation examples used for diagnostics), and puts them in separate archives.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--cmd", type=str, default = "run.pl",
                        help="Specifies the script to launch jobs."
                        " e.g. queue.pl for launching on SGE cluster run.pl"
                        " for launching on local machine")
    # feat options
    parser.add_argument("--feat.dir", type=str, dest='feat_dir', required = True,
                        help="Directory with features used for training the neural network.")
    parser.add_argument("--feat.online-ivector-dir", type=str, dest='online_ivector_dir',
                        default = None, action = train_lib.NullstrToNoneAction,
                        help="directory with the ivectors extracted in an online fashion.")
    parser.add_argument("--feat.cmvn-opts", type=str, dest='cmvn_opts',
                        default = None, action = train_lib.NullstrToNoneAction,
                        help="A string specifying '--norm-means' and '--norm-vars' values")

    # egs extraction options
    parser.add_argument("--frame-subsampling-factor", type=int, default=1,
                        help="Frames-per-second of features we train on."
                        " Divided by frames-per-second at output of the nnet3 model.")
    parser.add_argument("--frames-per-eg", type=int, default=8,
                        help="Number of frames of labels per example. "
                        "more->less disk space and less time preparing egs, "
                        "but more I/O during training. "
                        "note: the script may reduce this if reduce-frames-per-eg is true.")
    parser.add_argument("--left-context", type=int, default = 4,
                        help="Amount of left-context per eg (i.e. extra frames "
                        "of input features not present in the output supervision).")
    parser.add_argument("--right-context", type=int, default = 4,
                        help="Amount of right-context per eg")
    parser.add_argument("--valid-left-context", type=int, default = None,
                        help=" Amount of left-context for validation egs, typically"
                        " used in recurrent architectures to ensure matched"
                        " condition with training egs")
    parser.add_argument("--valid-right-context", type=int, default = None,
                        help=" Amount of right-context for validation egs, typically"
                        " used in recurrent architectures to ensure matched"
                        " condition with training egs")
    parser.add_argument("--compress-input", type=str, default = True,
                        action = train_lib.StrToBoolAction,
                        choices = ["true", "false"],
                        help="If false, disables compression. Might be necessary"
                        " to check if results will be affected.")
    parser.add_argument("--input-compress-format", type=int, default=0,
                        help="Format used for compressing the input features")

    parser.add_argument("--reduce-frames-per-eg", type=str, default = True,
                        action = train_lib.StrToBoolAction,
                        choices = ["true", "false"],
                        help = "If true, this script may reduce the frames-per-eg "
                        "if there is only one archive and even with the "
                        "reduced frames-per-eg, the number of "
                        "samples-per-iter that would result is less than or "
                        "equal to the user-specified value.")

    parser.add_argument("--num-utts-subset", type=int, default = 300,
                        help="Number of utterances in validation and training"
                        " subsets used for shrinkage and diagnostics")
    parser.add_argument("--num-utts-subset-valid", type=int,
                        help="Number of utterances in validation"
                        " subset used for diagnostics")
    parser.add_argument("--num-utts-subset-train", type=int,
                        help="Number of utterances in training"
                        " subset used for shrinkage and diagnostics")
    parser.add_argument("--num-train-egs-combine", type=int, default=10000,
                        help="Training examples for combination weights at the"
                        " very end.")
    parser.add_argument("--num-valid-egs-combine", type=int, default=0,
                        help="Validation examples for combination weights at the"
                        " very end.")
    parser.add_argument("--num-egs-diagnostic", type=int, default=4000,
                        help="Numer of frames for 'compute-probs' jobs")

    parser.add_argument("--samples-per-iter", type=int, default=400000,
                        help="This is the target number of egs in each archive of egs "
                        "(prior to merging egs).  We probably should have called "
                        "it egs_per_iter. This is just a guideline; it will pick "
                        "a number that divides the number of samples in the "
                        "entire data.")

    parser.add_argument("--stage", type=int, default=0,
                        help="Stage to start running script from")
    parser.add_argument("--num-jobs", type=int, default=6,
                        help="This should be set to the maximum number of jobs you are "
                        "comfortable to run in parallel; you can increase it if your disk "
                        "speed is greater and you have more machines.")
    parser.add_argument("--srand", type=int, default=0,
                        help="Rand seed for nnet3-copy-egs and nnet3-shuffle-egs")

    parser.add_argument("--targets-parameters", type=str, action='append',
                        required = True, dest = 'targets_para_array',
                        help = """Parameters for targets. Each set of parameters\n
                        corresponds to a separate output node of the neural\n
                        network. The targets can be sparse or dense.\n
                        The parameters used are:\n
                        --targets-rspecifier=<targets_rspecifier>   # rspecifier for the targets, can be alignment or matrix.\n
                        --num-targets=<n>   # targets dimension. required for sparse feats.\n
                        --target-type=<dense|sparse>""")

    parser.add_argument("--dir", type=str, required = True,
                        help="Directory to store the examples")

    print(' '.join(sys.argv))
    print(sys.argv)

    args = parser.parse_args()

    args = ProcessArgs(args)

    return args

def ProcessArgs(args):
    # process the options
    if args.num_utts_subset_valid is None:
        args.num_utts_subset_valid = args.num_utts_subset

    if args.num_utts_subset_train is None:
        args.num_utts_subset_train = args.num_utts_subset

    if args.valid_left_context is None:
        args.valid_left_context = args.left_context
    if args.valid_right_context is None:
        args.valid_right_context = args.right_context

    if ((args.left_context < 0) or (args.right_context < 0)
        or (args.valid_left_context < 0) or (args.valid_right_context < 0)):
        raise Exception("--{,valid-}{left,right}-context should be non-negative")

    return args

def CheckForRequiredFiles(feat_dir, targets_scps, online_ivector_dir = None):
    required_files = ['{0}/feats.scp'.format(feat_dir)]
    required_files.extend(targets_scps)
    if online_ivector_dir is not None:
        required_files.append('{0}/ivector_online.scp'.format(online_ivector_dir))
        required_files.append('{0}/ivector_period'.format(online_ivector_dir))

    for file in required_files:
        if not os.path.isfile(file):
            raise Exception('Expected {0} to exist.'.format(file))

def ParseTargetsParametersArray(para_array):
    targets_parser = argparse.ArgumentParser()
    targets_parser.add_argument("--output-name", type=str, required=True,
                                help = "Name of the output. e.g. output-xent")
    targets_parser.add_argument("--dim", type=int, default = -1,
                                help = "Target dimension (required for sparse targets")
    targets_parser.add_argument("--target-type", type=str, default = "dense",
                                choices = ["dense", "sparse"],
                                help = "Dense for matrix format")
    targets_parser.add_argument("--targets-scp", type=str, required=True,
                                help = "Scp file of targets; can be posteriors or matrices")
    targets_parser.add_argument("--compress", type=str, default=True,
                                action = train_lib.StrToBoolAction,
                                help = "Specifies whether the output must be compressed")
    targets_parser.add_argument("--compress-format", type=int, default = 0,
                                help = "Format for compressing target")
    targets_parser.add_argument("--deriv-weights-scp", type=str, default = "",
                                help = "Per-frame deriv weights for this output")
    targets_parser.add_argument("--scp2ark-cmd", type=str, default = "",
                                help = "The command that is used to convert targets scp to archive. e.g. An scp of alignments can be converted to posteriors using ali-to-post")

    targets_parameters = [ targets_parser.parse_args(shlex.split(x)) for x in para_array ]

    for t in targets_parameters:
        if (t.target_type == "dense"):
            dim = train_lib.GetFeatDimFromScp(t.targets_scp)
            if (t.dim != -1 and t.dim != dim):
                raise Exception('Mismatch in --dim provided and feat dim for file {0}; {1} vs {2}'.format(t.targets_scp, t.dim, dim))
            t.dim = -dim
        #if t.dim <= 0:
        #    raise ValueError("Expecting dim to be > 0 for output {0}".format(t.output_name))

    return targets_parameters

def SampleUtts(feat_dir, num_utts_subset, min_duration, exclude_list=None):
    utt2durs_dict = data_lib.GetUtt2Dur(feat_dir)
    utt2durs = utt2durs_dict.items()
    utt2uniq, uniq2utt = data_lib.GetUtt2Uniq(feat_dir)
    if num_utts_subset is None:
        num_utts_subset = len(utt2durs)
        if exclude_list is not None:
            num_utts_subset = num_utts_subset - len(exclude_list)

    random.shuffle(utt2durs)
    sampled_utts = []

    index = 0
    num_trials = 0
    while (len(sampled_utts) < num_utts_subset) and (num_trials <= len(utt2durs)):
        if utt2durs[index][-1] >= min_duration:
            if utt2uniq is not None:
                uniq_id = utt2uniq[utt2durs[index][0]]
                utts2add = uniq2utt[uniq_id]
            else:
                utts2add = [utt2durs[index][0]]
            exclude_utt = False
            if exclude_list is not None:
                for utt in utts2add:
                    if utt in exclude_list:
                        exclude_utt = True
                        break
            if not exclude_utt:
                for utt in utts2add:
                    sampled_utts.append(utt)

            index = index + 1
        num_trials = num_trials + 1
    if exclude_list is not None:
        assert(len(set(exclude_list).intersection(sampled_utts)) == 0)
    if len(sampled_utts) < num_utts_subset:
        raise Exception("Number of utterances which have duration of at least "
                "{md} seconds is really low (required={rl}, available={al}). Please check your data.".format(md = min_duration, al=len(sampled_utts), rl=num_utts_subset))
    sampled_utts_durs = []
    for utt in sampled_utts:
        sampled_utts_durs.append([utt, utt2durs_dict[utt]])
    return sampled_utts, sampled_utts_durs

def WriteList(listd, file_name):
    file_handle = open(file_name, 'w')
    for item in listd:
        file_handle.write(str(item)+"\n")
    file_handle.close()

def GetMaxOpenFiles():
    stdout, stderr = train_lib.RunKaldiCommand("ulimit -n")
    return int(stdout)

def GetFeatIvectorStrings(dir, feat_dir, split_feat_dir, cmvn_opt_string, ivector_dir = None):

    train_feats = "ark,s,cs:utils/filter_scp.pl --exclude {dir}/valid_uttlist {sdir}/JOB/feats.scp | apply-cmvn {cmvn} --utt2spk=ark:{sdir}/JOB/utt2spk scp:{sdir}/JOB/cmvn.scp scp:- ark:- |".format(dir = dir, sdir = split_feat_dir, cmvn = cmvn_opt_string)
    valid_feats="ark,s,cs:utils/filter_scp.pl {dir}/valid_uttlist {fdir}/feats.scp | apply-cmvn {cmvn} --utt2spk=ark:{fdir}/utt2spk scp:{fdir}/cmvn.scp scp:- ark:- |".format(dir = dir, fdir = feat_dir, cmvn = cmvn_opt_string)
    train_subset_feats="ark,s,cs:utils/filter_scp.pl {dir}/train_subset_uttlist  {fdir}/feats.scp | apply-cmvn {cmvn} --utt2spk=ark:{fdir}/utt2spk scp:{fdir}/cmvn.scp scp:- ark:- |".format(dir = dir, fdir = feat_dir, cmvn = cmvn_opt_string)
    feats_subset_func = lambda subset_list : "ark,s,cs:utils/filter_scp.pl {subset_list} {fdir}/feats.scp | apply-cmvn {cmvn} --utt2spk=ark:{fdir}/utt2spk scp:{fdir}/cmvn.scp scp:- ark:- |".format(dir = dir, subset_list = subset_list, fdir = feat_dir, cmvn = cmvn_opt_string)

    if ivector_dir is not None:
        ivector_period = train_lib.GetIvectorPeriod(ivector_dir)
        ivector_opt="--ivectors='ark,s,cs:utils/filter_scp.pl {sdir}/JOB/utt2spk {idir}/ivector_online.scp | subsample-feats --n=-{period} scp:- ark:- |'".format(sdir = split_feat_dir, idir = ivector_dir, period = ivector_period)
        valid_ivector_opt="--ivectors='ark,s,cs:utils/filter_scp.pl {dir}/valid_uttlist {idir}/ivector_online.scp | subsample-feats --n=-{period} scp:- ark:- |'".format(dir = dir, idir = ivector_dir, period = ivector_period)
        train_subset_ivector_opt="--ivectors='ark,s,cs:utils/filter_scp.pl {dir}/train_subset_uttlist {idir}/ivector_online.scp | subsample-feats --n=-{period} scp:- ark:- |'".format(dir = dir, idir = ivector_dir, period = ivector_period)
    else:
        ivector_opt = ''
        valid_ivector_opt = ''
        train_subset_ivector_opt = ''

    return {'train_feats':train_feats,
            'valid_feats':valid_feats,
            'train_subset_feats':train_subset_feats,
            'feats_subset_func':feats_subset_func,
            'ivector_opts':ivector_opt,
            'valid_ivector_opts':valid_ivector_opt,
            'train_subset_ivector_opts':train_subset_ivector_opt,
            'feat_dim':train_lib.GetFeatDim(feat_dir),
            'ivector_dim':train_lib.GetIvectorDim(ivector_dir)}

def GetEgsOptions(targets_parameters, frames_per_eg,
                  left_context, right_context,
                  valid_left_context, valid_right_context,
                  frame_subsampling_factor, compress_input,
                  input_compress_format = 0, length_tolerance = 0):
    # TODO: Make use of frame_subsampling_factor

    train_egs_opts = "--left-context={lc} --right-context={rc} --num-frames={n} --compress-input={comp} --input-compress-format={icf} --compress-targets={ct} --targets-compress-formats={tcf} --length-tolerance={tol} --output-names={names} --output-dims={dims}".format(lc = left_context, rc = right_context,
              n = frames_per_eg, comp = compress_input, icf = input_compress_format,
              ct = ':'.join([ "true" for t in targets_parameters if t.compress else "false" ]),
              tcf = ':'.join([ str(t.compress_format) for t in targets_parameters ]),
              tol = length_tolerance,
              names = ':'.join([ t.output_name for t in targets_parameters ]),
              dims = ':'.join([ str(t.dim) for t in targets_parameters ])
              )

    valid_egs_opts = "--left-context={vlc} --right-context={vrc} --num-frames={n} --compress-input={comp} --input-compress-format={icf} --compress-targets={ct} --targets-compress-formats={tcf} --length-tolerance={tol} --output-names={names} --output-dims={dims}".format(vlc = valid_left_context,
              vrc = valid_right_context, n = frames_per_eg, comp = compress_input, icf = input_compress_format,
              ct = ':'.join([ "true" for t in targets_parameters if t.compress else "false" ]),
              tcf = ':'.join([ str(t.compress_format) for t in targets_parameters ]),
              tol = length_tolerance,
              names = ':'.join([ t.output_name for t in targets_parameters ]),
              dims = ':'.join([ str(t.dim) for t in targets_parameters ])
              )

    return {'train_egs_opts' : train_egs_opts,
            'valid_egs_opts' : valid_egs_opts}

def GetTargetsList(targets_parameters, subset_list):
    targets_list = ""
    n = 0
    for t in targets_parameters:
        n += 1
        rspecifier = "ark,s,cs:" if t.scp2ark_cmd != "" else "scp,s,cs:"
        rspecifier += GetSubsetRspecifier(t.targets_scp, subset_list)
        rspecifier += t.scp2ark_cmd
        deriv_weights_rspecifier = ""
        if t.deriv_weights_scp != "":
            deriv_weights_rspecifier = "scp,s,cs:"
            deriv_weights_rspecifier += GetSubsetRspecifier(t.deriv_weights_scp, subset_list)
        this_targets = '''"{rspecifier}" "{dw}"'''.format(rspecifier = rspecifier, dw = deriv_weights_rspecifier)

        if n == 1:
            targets_list = this_targets
        else:
            targets_list += " " + this_targets
    return targets_list

def GetSubsetRspecifier(scp_file, subset_list):
    if scp_file == "":
        return ""
    return "utils/filter_scp.pl {subset} {scp} |".format(subset = subset_list, scp = scp_file)

def SplitScp(scp_file, num_jobs):
    out_scps = [ "{0}.{1}".format(scp_file, n) for n in range(1, num_jobs + 1) ]
    train_lib.RunKaldiCommand("utils/split_scp.pl {scp} {oscps}".format(
                              scp = scp_file,
                              oscps = ' '.join(out_scps)))
    return out_scps

def GenerateValidTrainSubsetEgs(dir, targets_parameters,
                                feat_ivector_strings, egs_opts,
                                num_train_egs_combine,
                                num_valid_egs_combine,
                                num_egs_diagnostic, cmd,
                                num_jobs = 1):
    wait_pids = []

    logger.info("Creating validation and train subset examples.")

    SplitScp('{0}/valid_uttlist'.format(dir), num_jobs)
    SplitScp('{0}/train_subset_uttlist'.format(dir), num_jobs)

    valid_pid = train_lib.RunKaldiCommand("""
  {cmd} JOB=1:{nj} {dir}/log/create_valid_subset.JOB.log \
          nnet3-get-egs-multiple-targets {v_iv_opt} {v_egs_opt} "{v_feats}" {targets} ark:{dir}/valid_all.JOB.egs""".format(
          cmd = cmd, nj = num_jobs, dir = dir,
          v_egs_opt = egs_opts['valid_egs_opts'],
          v_iv_opt = feat_ivector_strings['valid_ivector_opts'],
          v_feats = feat_ivector_strings['feats_subset_func']('{dir}/valid_uttlist.JOB'.format(dir=dir)),
          targets = GetTargetsList(targets_parameters, '{dir}/valid_uttlist.JOB'.format(dir=dir)) ), wait = False)

    train_pid = train_lib.RunKaldiCommand("""
  {cmd} JOB=1:{nj} {dir}/log/create_train_subset.JOB.log \
          nnet3-get-egs-multiple-targets {t_iv_opt} {v_egs_opt} "{t_feats}" {targets} ark:{dir}/train_subset_all.JOB.egs""".format(
          cmd = cmd, nj = num_jobs, dir = dir,
          v_egs_opt = egs_opts['valid_egs_opts'],
          t_iv_opt = feat_ivector_strings['train_subset_ivector_opts'],
          t_feats = feat_ivector_strings['feats_subset_func']('{dir}/train_subset_uttlist.JOB'.format(dir=dir)),
          targets = GetTargetsList(targets_parameters, '{dir}/train_subset_uttlist.JOB'.format(dir=dir)) ), wait = False)

    wait_pids.append(valid_pid)
    wait_pids.append(train_pid)

    for pid in wait_pids:
        stdout, stderr = pid.communicate()
        if pid.returncode != 0:
            raise Exception(stderr)

    valid_egs_all = ' '.join([ '{dir}/valid_all.{n}.egs'.format(dir=dir, n=n) for n in range(1, num_jobs + 1) ])
    train_subset_egs_all = ' '.join([ '{dir}/train_subset_all.{n}.egs'.format(dir=dir, n=n) for n in range(1, num_jobs + 1) ])

    wait_pids = []
    logger.info("... Getting subsets of validation examples for diagnostics and combination.")
    pid = train_lib.RunKaldiCommand("""
  {cmd} {dir}/log/create_valid_subset_combine.log \
    cat {valid_egs_all} \| nnet3-subset-egs --n={nve_combine} ark:- \
    ark:{dir}/valid_combine.egs""".format(
        cmd = cmd, dir = dir, valid_egs_all = valid_egs_all,
        nve_combine = num_valid_egs_combine), wait = False)
    wait_pids.append(pid)

    pid = train_lib.RunKaldiCommand("""
  {cmd} {dir}/log/create_valid_subset_diagnostic.log \
    cat {valid_egs_all} \| nnet3-subset-egs --n={ne_diagnostic} ark:- \
    ark:{dir}/valid_diagnostic.egs""".format(
        cmd = cmd, dir = dir, valid_egs_all = valid_egs_all,
        ne_diagnostic = num_egs_diagnostic), wait = False)
    wait_pids.append(pid)

    pid = train_lib.RunKaldiCommand("""
  {cmd} {dir}/log/create_train_subset_combine.log \
    cat {train_subset_egs_all} \| nnet3-subset-egs --n={nte_combine} ark:- \
    ark:{dir}/train_combine.egs""".format(
        cmd = cmd, dir = dir, train_subset_egs_all = train_subset_egs_all,
        nte_combine = num_train_egs_combine), wait = False)
    wait_pids.append(pid)

    pid = train_lib.RunKaldiCommand("""
  {cmd} {dir}/log/create_train_subset_diagnostic.log \
    cat {train_subset_egs_all} \| nnet3-subset-egs --n={ne_diagnostic} ark:- \
    ark:{dir}/train_diagnostic.egs""".format(
        cmd = cmd, dir = dir, train_subset_egs_all = train_subset_egs_all,
        ne_diagnostic = num_egs_diagnostic), wait = False)
    wait_pids.append(pid)

    for pid in wait_pids:
        stdout, stderr = pid.communicate()
        if pid.returncode != 0:
            raise Exception(stderr)

    train_lib.RunKaldiCommand(""" cat {dir}/valid_combine.egs {dir}/train_combine.egs > {dir}/combine.egs""".format(dir = dir))

    # perform checks
    for file_name in '{0}/combine.egs {0}/train_diagnostic.egs {0}/valid_diagnostic.egs'.format(dir).split():
        if os.path.getsize(file_name) == 0:
            raise Exception("No examples in {0}".format(file_name))

    # clean-up
    for x in '{0}/valid_all.*.egs {0}/train_subset_all.*.egs {0}/train_combine.egs {0}/valid_combine.egs'.format(dir).split():
        for file_name in glob.glob(x):
            os.remove(file_name)

def GenerateTrainingExamplesInternal(dir, targets_parameters, feat_dir,
                                     train_feats_string, train_egs_opts_string,
                                     ivector_opts,
                                     num_jobs, frames_per_eg,
                                     samples_per_iter, cmd, srand = 0,
                                     reduce_frames_per_eg = True,
                                     only_shuffle = False,
                                     dry_run = False):

    # The examples will go round-robin to egs_list.  Note: we omit the
    # 'normalization.fst' argument while creating temporary egs: the phase of egs
    # preparation that involves the normalization FST is quite CPU-intensive and
    # it's more convenient to do it later, in the 'shuffle' stage.  Otherwise to
    # make it efficient we need to use a large 'nj', like 40, and in that case
    # there can be too many small files to deal with, because the total number of
    # files is the product of 'nj' by 'num_archives_intermediate', which might be
    # quite large.
    num_frames = data_lib.GetNumFrames(feat_dir)
    num_archives = (num_frames) / (frames_per_eg * samples_per_iter) + 1

    reduced = False
    while (reduce_frames_per_eg and frames_per_eg > 1 and
            num_frames / ((frames_per_eg-1)*samples_per_iter) == 0):
        frames_per_eg -= 1
        num_archives = 1
        reduced = True

    if reduced:
        logger.info("Reduced frames-per-eg to {0} because amount of data is small".format(frames_per_eg))

    max_open_files = GetMaxOpenFiles()
    num_archives_intermediate = num_archives
    archives_multiple = 1
    while (num_archives_intermediate+4) > max_open_files:
      archives_multiple = archives_multiple + 1
      num_archives_intermediate = int(math.ceil(float(num_archives) / archives_multiple))
    num_archives = num_archives_intermediate * archives_multiple
    egs_per_archive = num_frames/(frames_per_eg * num_archives)

    if egs_per_archive > samples_per_iter:
        raise Exception("egs_per_archive({epa}) > samples_per_iter({fpi}). This is an error in the logic for determining egs_per_archive".format(epa = egs_per_archive, fpi = samples_per_iter))

    if dry_run:
        Cleanup(dir, archives_multiple)
        return {'num_frames':num_frames,
                'num_archives':num_archives,
                'egs_per_archive':egs_per_archive}

    logger.info("Splitting a total of {nf} frames into {na} archives, each with {epa} egs.".format(nf = num_frames, na = num_archives, epa = egs_per_archive))

    if os.path.isdir('{0}/storage'.format(dir)):
        # this is a striped directory, so create the softlinks
        data_lib.CreateDataLinks(["{dir}/egs.{x}.ark".format(dir = dir, x = x) for x in range(1, num_archives + 1)])
        for x in range(1, num_archives_intermediate + 1):
            data_lib.CreateDataLinks(["{dir}/egs_orig.{y}.{x}.ark".format(dir = dir, x = x, y = y) for y in range(1, num_jobs + 1)])

    split_feat_dir = "{0}/split{1}".format(feat_dir, num_jobs)
    egs_list = ' '.join(['ark:{dir}/egs_orig.JOB.{ark_num}.ark'.format(dir=dir, ark_num = x) for x in range(1, num_archives_intermediate + 1)])
    if not only_shuffle:
        train_lib.RunKaldiCommand("""
        {cmd} JOB=1:{nj} {dir}/log/get_egs.JOB.log \
        nnet3-get-egs-multiple-targets {iv_opts} {egs_opts} \
         "{feats}" {targets} ark:- \| \
        nnet3-copy-egs --random=true --srand=$[JOB+{srand}] ark:- {egs_list}""".format(
            cmd = cmd, nj = num_jobs, dir = dir, srand = srand,
            iv_opts = ivector_opts, egs_opts = train_egs_opts_string,
            feats = train_feats_string,
            targets = GetTargetsList(targets_parameters, '{sdir}/JOB/utt2spk'.format(sdir=split_feat_dir)),
            egs_list = egs_list))

    logger.info("Recombining and shuffling order of archives on disk")
    egs_list = ' '.join(['{dir}/egs_orig.{n}.JOB.ark'.format(dir=dir, n = x) for x in range(1, num_jobs + 1)])
    output_list = []
    if archives_multiple == 1:
        # there are no intermediate archives so just shuffle egs across
        # jobs and dump them into a single output
        train_lib.RunKaldiCommand("""
    {cmd} --max-jobs-run {msjr} JOB=1:{nai} {dir}/log/shuffle.JOB.log \
      nnet3-shuffle-egs --srand=$[JOB+{srand}] "ark:cat {egs_list}|" ark:{dir}/egs.JOB.ark""".format(
              cmd = cmd, msjr = num_jobs,
              nai = num_archives_intermediate, srand = srand,
              dir = dir, egs_list = egs_list))
    else:
        # there are intermediate archives so we shuffle egs across jobs
        # and split them into archives_multiple output archives
        output_archives = ' '.join(["ark:{dir}/egs.JOB.{ark_num}.ark".format(dir = dir, ark_num = x) for x in range(1, archives_multiple + 1)])
        # archives were created as egs.x.y.ark
        # linking them to egs.i.ark format which is expected by the training
        # scripts
        for i in range(1, num_archives_intermediate + 1):
            for j in range(1, archives_multiple + 1):
                archive_index = (i-1) * archives_multiple + j
                ForceSymLink("egs.{0}.ark".format(archive_index),
                             "{dir}/egs.{i}.{j}.ark".format(dir = dir, i = i, j = j))

        train_lib.RunKaldiCommand("""
    {cmd} --max-jobs-run {msjr} JOB=1:{nai} {dir}/log/shuffle.JOB.log \
      nnet3-shuffle-egs --srand=$[JOB+{srand}] "ark:cat {egs_list}|" ark:- \| \
      nnet3-copy-egs ark:- {oarks}""".format(
          cmd = cmd, msjr = num_jobs,
          nai = num_archives_intermediate, srand = srand,
          dir = dir, egs_list = egs_list, oarks = output_archives))

    Cleanup(dir, archives_multiple)
    return {'num_frames':num_frames,
            'num_archives':num_archives,
            'egs_per_archive':egs_per_archive}

import os, errno
def ForceSymLink(source, dest):
    try:
        os.symlink(source, dest)
    except OSError, e:
        if e.errno == errno.EEXIST:
            os.remove(dest)
            os.symlink(source, dest)
        else:
            raise e

def Cleanup(dir, archives_multiple):
    logger.info("Removing temporary archives in {0}.".format(dir))
    for file_name in glob.glob("{0}/egs_orig*".format(dir)):
        real_path = os.path.realpath(file_name)
        data_lib.TryToDelete(real_path)
        data_lib.TryToDelete(file_name)

    if archives_multiple > 1:
        # there will be some extra soft links we want to delete
        for file_name in glob.glob('{0}/egs.*.*.ark'.format(dir)):
            os.remove(file_name)

def CreateDirectory(dir):
    import os, errno
    try:
        os.makedirs(dir)
    except OSError, e:
        if e.errno == errno.EEXIST:
            pass

def GenerateTrainingExamples(dir, targets_parameters, feat_dir,
                             feat_ivector_strings, egs_opts,
                             frame_shift, frames_per_eg, samples_per_iter,
                             cmd, num_jobs, srand = 0,
                             only_shuffle = False, dry_run = False):

    # generate the training options string with the given chunk_width
    train_egs_opts = egs_opts['train_egs_opts']
    # generate the feature vector string with the utt list for the
    # current chunk width
    train_feats = feat_ivector_strings['train_feats']

    if os.path.isdir('{0}/storage'.format(dir)):
        real_paths = [os.path.realpath(x).strip("/") for x in glob.glob('{0}/storage/*'.format(dir))]
        train_lib.RunKaldiCommand("""
            utils/create_split_dir.pl {target_dirs} {dir}/storage""".format(target_dirs = " ".join(real_paths), dir = dir))

    info = GenerateTrainingExamplesInternal(dir, targets_parameters, feat_dir,
                                            train_feats, train_egs_opts,
                                            feat_ivector_strings['ivector_opts'],
                                            num_jobs, frames_per_eg,
                                            samples_per_iter, cmd,
                                            srand = srand,
                                            only_shuffle = only_shuffle,
                                            dry_run = dry_run)

    return info

def WriteEgsInfo(info, info_dir):
    for x in ['num_frames','num_archives','egs_per_archive',
              'feat_dim','ivector_dim',
              'left_context','right_context','frames_per_eg']:
        WriteList([info['{0}'.format(x)]], '{0}/{1}'.format(info_dir, x))

def GenerateEgs(egs_dir, feat_dir, targets_para_array,
                online_ivector_dir = None,
                frames_per_eg = 8,
                left_context = 4,
                right_context = 4,
                valid_left_context = None,
                valid_right_context = None,
                cmd = "run.pl", stage = 0,
                cmvn_opts = None,
                compress_input = True,
                input_compress_format = 0,
                num_utts_subset = 300,
                num_train_egs_combine = 1000,
                num_valid_egs_combine = 0,
                num_egs_diagnostic = 4000,
                samples_per_iter = 400000,
                num_jobs = 6,
                frame_subsampling_factor = 1,
                srand = 0):

    for directory in '{0}/log {0}/info'.format(egs_dir).split():
            CreateDirectory(directory)

    WriteList(cmvn_opts if cmvn_opts is not None else '', '{0}/cmvn_opts'.format(egs_dir))

    targets_parameters = ParseTargetsParametersArray(targets_para_array)

    # Check files
    CheckForRequiredFiles(feat_dir,
                          [t.targets_scp for t in targets_parameters],
                          online_ivector_dir)

    frame_shift = data_lib.GetFrameShift(feat_dir)
    min_duration = frames_per_eg * frame_shift;
    valid_utts = SampleUtts(feat_dir, num_utts_subset, min_duration)[0]
    train_subset_utts = SampleUtts(feat_dir, num_utts_subset, min_duration, exclude_list = valid_utts)[0]
    train_utts, train_utts_durs = SampleUtts(feat_dir, None, -1, exclude_list = valid_utts)

    WriteList(valid_utts, '{0}/valid_uttlist'.format(egs_dir))
    WriteList(train_subset_utts, '{0}/train_subset_uttlist'.format(egs_dir))
    WriteList(train_utts, '{0}/train_uttlist'.format(egs_dir))


    # split the training data into parts for individual jobs
    # we will use the same number of jobs as that used for alignment
    split_feat_dir = train_lib.SplitData(feat_dir, num_jobs)
    feat_ivector_strings = GetFeatIvectorStrings(egs_dir, feat_dir,
            split_feat_dir, cmvn_opts, ivector_dir = online_ivector_dir)

    egs_opts = GetEgsOptions(targets_parameters = targets_parameters,
                             frames_per_eg = frames_per_eg,
                             left_context = left_context, right_context = right_context,
                             valid_left_context = valid_left_context, valid_right_context = valid_right_context,
                             frame_subsampling_factor = frame_subsampling_factor,
                             compress_input = compress_input, input_compress_format = input_compress_format)

    if stage <= 2:
        logger.info("Generating validation and training subset examples")

        GenerateValidTrainSubsetEgs(egs_dir, targets_parameters,
                                    feat_ivector_strings, egs_opts,
                                    num_train_egs_combine,
                                    num_valid_egs_combine,
                                    num_egs_diagnostic, cmd,
                                    num_jobs = num_jobs)

    logger.info("Generating training examples on disk.")
    info = GenerateTrainingExamples(dir = egs_dir,
                             targets_parameters = targets_parameters,
                             feat_dir = feat_dir,
                             feat_ivector_strings = feat_ivector_strings,
                             egs_opts = egs_opts,
                             frame_shift = frame_shift,
                             frames_per_eg = frames_per_eg,
                             samples_per_iter = samples_per_iter,
                             cmd = cmd,
                             num_jobs = num_jobs,
                             srand = srand,
                             only_shuffle = True if stage > 3 else False,
                             dry_run = True if stage > 4 else False)

    info['feat_dim'] = feat_ivector_strings['feat_dim']
    info['ivector_dim'] = feat_ivector_strings['ivector_dim']
    info['left_context'] = left_context
    info['right_context'] = right_context
    info['frames_per_eg'] = frames_per_eg

    WriteEgsInfo(info, '{dir}/info'.format(dir=egs_dir))

def Main():
    args = GetArgs()
    GenerateEgs(args.dir, args.feat_dir, args.targets_para_array,
                     online_ivector_dir = args.online_ivector_dir,
                     frames_per_eg = args.frames_per_eg,
                     left_context = args.left_context,
                     right_context = args.right_context,
                     valid_left_context = args.valid_left_context,
                     valid_right_context = args.valid_right_context,
                     cmd = args.cmd, stage = args.stage,
                     cmvn_opts = args.cmvn_opts,
                     compress_input = args.compress_input,
                     input_compress_format = args.input_compress_format,
                     num_utts_subset = args.num_utts_subset,
                     num_train_egs_combine = args.num_train_egs_combine,
                     num_valid_egs_combine = args.num_valid_egs_combine,
                     num_egs_diagnostic = args.num_egs_diagnostic,
                     samples_per_iter = args.samples_per_iter,
                     num_jobs = args.num_jobs,
                     frame_subsampling_factor = args.frame_subsampling_factor,
                     srand = args.srand)

if __name__ == "__main__":
    Main()

