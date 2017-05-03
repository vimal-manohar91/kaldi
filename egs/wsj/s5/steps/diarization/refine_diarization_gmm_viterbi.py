#! /usr/bin/env python

# Copyright 2016    Vijayaditya Peddinti.
#           2016    Vimal Manohar
# Apache 2.0.

""" This script refines a diarization clustering using HMM-GMMs per-recording.
"""

from __future__ import print_function
import argparse
import logging
import multiprocessing
import os
import sys

sys.path.insert(0, 'steps')
import libs.common as common_lib

logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting resegmentation')


def get_args():
    """ Get args from stdin.
    """

    parser = argparse.ArgumentParser(
        description="""Refine a clustering using HMM-GMM.""")

    parser.add_argument("--nj", type=int, default=40,
                        help="Number of recordings to process in parallel.")
    parser.add_argument("--utt-nj", type=int, default=80,
                        help="Number of utterances to process in parallel.")
    parser.add_argument("--cmd", type=str, default="queue.pl",
                        help="Command to run jobs")
    parser.add_argument("--stage", type=int, default=-10,
                        help="Run script from this stage")
    parser.add_argument("--num-threads", type=int, default=8,
                        help="Number of threads")

    parser.add_argument("--num-iters", type=int, default=5,
                        help="Number of iterations of Viterbi resegmentation.")
    parser.add_argument("--max-iter-inc", type=int, default=3,
                        help="Number of iterations to increase Gaussians.")
    parser.add_argument("--num-gauss-per-cluster", type=float, default=5,
                        help="Average number of Gaussians per cluster")

    parser.add_argument("--beam", type=int, default=10,
                        help="Beam for decoding.")
    parser.add_argument("--retry-beam", type=int, default=40,
                        help="Beam for decoding for retry when alignment "
                        "fails")
    parser.add_argument("--max-active", type=int, default=1000,
                        help="Maximum number of active states during decoding")
    parser.add_argument("--power", type=float, default=0.25,
                        help="Exponent for number of gaussians according to "
                        "occurence counts")
    parser.add_argument("--update-opts", type=str, default="",
                        help="Other options for GMM ML-update")
    parser.add_argument("--transition-scale", type=float, default=1.0,
                        help="""Scale on transition probabilities relative to
                        acoustics""")
    parser.add_argument("--self-loop-scale", type=float, default=0.1,
                        help="""Scale on self-loop probabilities relative to
                        acoustics""")
    parser.add_argument("--acoustic-scale", type=float, default=0.1,
                        help="""Scale on acoustic likelihoods relative to graph
                        scores""")

    parser.add_argument("--transition-prob", type=float, default=0.9,
                        help="""Transition probability on the HMM""")
    parser.add_argument("--self-loop-prob", type=float, default=0.1,
                        help="""Self-loop probability on the HMM""")

    parser.add_argument("--data", required=True,
                        help="Clustered data directory.")
    parser.add_argument("--dir", required=True,
                        help="Working directory to store models etc.")
    parser.add_argument("--out-data", required=True,
                        help="Output data directory")

    args = parser.parse_args()
    args = check_args(args)

    return args


def check_args(args):
    """Check values of arguments.
    """

    if args.max_iter_inc > args.num_iters - 2:
        args.max_iter_inc = args.num_iters - 2

    if args.num_iters < 1:
        raise TypeError("--num-iters expected to be > 1")

    return args


def check_files(args):
    """Check if required files exist"""
    for f in "{data}/spk2utt {data}/feats.scp".format(data=args.data).split():
        if not os.path.exists(f):
            logger.error("Could not find file {0}".format(f))
            raise SystemExit(1)


def write_topo(args, num_clusters, topo_file):
    """Write a HMM topology with <num-clusters> clusters into a <topo-file>.
    A "cluster" takes the place of a "phone" in the ASR HMM.
    A one-state topology is used with a self loop on it with
    self-loop-prob and transition-prob fixed from the input args.
    """
    print ("<Topology>", file=topo_file)
    print ("<TopologyEntry>", file=topo_file)
    print ("<ForPhones>", file=topo_file)
    print (" ".join([str(x) for x in range(1, num_clusters + 1)]),
           file=topo_file)
    print ("</ForPhones>", file=topo_file)
    print ("<State> 0 <PdfClass> 0 <Transition> 0 {0} "
           "<Transition> 1 {1} </State>".format(
               args.self_loop_prob, args.transition_prob),
           file=topo_file)
    print ("<State> 1 </State>", file=topo_file)
    print ("</TopologyEntry>", file=topo_file)
    print ("</Topology>", file=topo_file)


def run_align_iter(args, reco_dir, iter_, num_gauss):
    """Do one iteration of Viterbi alignment and ML-update of HMM-GMM.

    Arguments:
        reco_dir -- Output directory corresponding to the recording
        iter_ -- Current iteration number
        num_gauss -- The maximum number Gaussians allowed in the HMM-GMM.
    """
    logger.info("Align pass ({iter}) -- "
                "num-gauss={num_gauss}".format(
                    iter=iter_, num_gauss=num_gauss))

    align_opts = []
    align_opts.append(
        "--transition-scale={0}".format(args.transition_scale))
    align_opts.append(
        "--self-loop-scale={0}".format(args.self_loop_scale))
    align_opts.append(
        "--acoustic-scale={0}".format(args.acoustic_scale))
    align_opts.append("--beam={0}".format(args.beam))
    align_opts.append("--retry-beam={0}".format(args.retry_beam))

    feats_rspecifier = (
        "ark,s,cs:utils/filter_scp.pl {reco_dir}/text.JOB.{nj} "
        "{data}/feats.scp | add-deltas scp:- ark:- |"
        "".format(reco_dir=reco_dir, nj=args.utt_nj, data=args.data))

    common_lib.run_job(
        """{cmd} JOB=1:{nj} {reco_dir}/log/align.{iter}.JOB.log """
        """gmm-align-compiled {align_opts} """
        """{reco_dir}/{iter}.mdl """
        """ "ark:gunzip -c {reco_dir}/fsts.JOB.gz |" "{feats}" """
        """ "ark:| gzip -c > {reco_dir}/ali.{iter}.JOB.gz" """
        """ """.format(cmd=args.cmd, nj=args.utt_nj,
                       reco_dir=reco_dir, iter=iter_,
                       align_opts=" ".join(align_opts),
                       feats=feats_rspecifier))

    common_lib.run_job(
        """{cmd} JOB=1:{nj} {reco_dir}/log/acc.{iter}.JOB.log """
        """gmm-acc-stats-ali {reco_dir}/{iter}.mdl "{feats}" """
        """ "ark,s,cs:gunzip -c {reco_dir}/ali.{iter}.JOB.gz |" """
        """ {reco_dir}/{iter}.JOB.acc"""
        """ """.format(cmd=args.cmd, nj=args.utt_nj,
                       reco_dir=reco_dir, iter=iter_,
                       feats=feats_rspecifier))

    logger.info("Estimate pass ({iter})".format(iter=iter_))
    common_lib.run_job(
        """{cmd} {reco_dir}/log/update.{iter}.log """
        """gmm-est --mix-up={num_gauss} --power={power} """
        """--update-flags='mvwt' {update_opts} """
        """{reco_dir}/{iter}.mdl """
        """ "gmm-sum-accs - {reco_dir}/{iter}.*.acc |" """
        """{reco_dir}/{next_iter}.mdl"""
        """ """.format(cmd=args.cmd, reco_dir=reco_dir,
                       iter=iter_, next_iter=iter_ + 1,
                       power=args.power, num_gauss=num_gauss,
                       update_opts=args.update_opts))


def run_decode_iter(args, reco_dir, iter_, num_gauss):
    """Do one iteration of EM update of HMM-GMM using lattice posterior stats.

    Arguments:
        reco_dir -- Output directory corresponding to the recording
        iter_ -- Current iteration number
        num_gauss -- The maximum number Gaussians allowed in the HMM-GMM.
    """
    logger.info("Decode pass ({iter}) -- "
                "num-gauss={num_gauss}".format(
                    iter=iter_, num_gauss=num_gauss))
    decode_opts = []
    decode_opts.append("--beam={0}".format(args.beam))
    decode_opts.append("--max-active={0}".format(args.max_active))
    decode_opts.append(
        "--acoustic-scale={0}".format(args.acoustic_scale))
    decode_opts.append(
        "--word-determinize=false --phone-determinize=false "
        "--minimize=false")

    feats_rspecifier = (
        "ark,s,cs:utils/filter_scp.pl {reco_dir}/text.JOB.{nj} "
        "{data}/feats.scp | add-deltas scp:- ark:- |"
        "".format(reco_dir=reco_dir, nj=args.utt_nj, data=args.data))

    common_lib.run_job(
        """{cmd} JOB=1:{nj} {reco_dir}/log/decode.{iter}.JOB.log """
        """gmm-latgen-faster {decode_opts} """
        """ {reco_dir}/{iter}.mdl {reco_dir}/HCLG.fst """
        """ "{feats}" """
        """ "ark:| gzip -c > {reco_dir}/lat.{iter}.JOB.gz" """
        """ """.format(cmd=args.cmd, nj=args.utt_nj,
                       reco_dir=reco_dir, iter=iter_,
                       decode_opts=" ".join(decode_opts),
                       feats=feats_rspecifier))

    common_lib.run_job(
        """{cmd} JOB=1:{nj} """
        """{reco_dir}/log/lattice_to_post.{iter}.JOB.log """
        """lattice-to-post --acoustic-scale={acwt} """
        """ "ark:gunzip -c {reco_dir}/lat.{iter}.JOB.gz |" """
        """ "ark:| gzip -c > {reco_dir}/post.{iter}.JOB.gz" """
        """ """.format(cmd=args.cmd, nj=args.utt_nj,
                       reco_dir=reco_dir, iter=iter_,
                       acwt=args.acoustic_scale))

    common_lib.run_job(
        """{cmd} JOB=1:{nj} {reco_dir}/log/acc.{iter}.JOB.log """
        """gmm-acc-stats {reco_dir}/{iter}.mdl "{feats}" """
        """ "ark,s,cs:gunzip -c {reco_dir}/post.{iter}.JOB.gz |" """
        """ {reco_dir}/{iter}.JOB.acc"""
        """ """.format(cmd=args.cmd, nj=args.utt_nj,
                       reco_dir=reco_dir, iter=iter_,
                       feats=feats_rspecifier))

    logger.info("Estimate pass ({iter})".format(iter=iter_))
    common_lib.run_job(
        """{cmd} {reco_dir}/log/update.{iter}.log """
        """gmm-est --update-flags='mvw' {update_opts} """
        """{reco_dir}/{iter}.mdl """
        """ "gmm-sum-accs - {reco_dir}/{iter}.*.acc |" """
        """{reco_dir}/{next_iter}.mdl"""
        """ """.format(cmd=args.cmd, reco_dir=reco_dir,
                       iter=iter_, next_iter=iter_ + 1,
                       update_opts=args.update_opts))


def run_per_reco(args, feat_dim, utt2spk, reco, utts):
    """Run Viterbi for a recording.
    """

    sorted_clusters = sorted(list(set([utt2spk[utt] for utt in utts])))
    cluster2int = {}
    for i, cluster in enumerate(sorted_clusters):
        cluster2int[cluster] = i + 1

    num_clusters = len(sorted_clusters)
    logger.info("For recording {reco}, got {num_clusters} clusters".format(
        reco=reco, num_clusters=num_clusters))

    reco_dir = "{0}/refine_{1}".format(args.dir, reco)
    if os.path.exists("{reco_dir}/.done".format(reco_dir=reco_dir)):
        logger.info("{0}/.done exits! Skipping {1}".format(reco_dir, reco))
        return

    if not os.path.exists(reco_dir):
        os.makedirs(reco_dir)

    logger.info("For recording {reco}, initializing GMM".format(reco=reco))
    if args.stage <= -3:
        topo_file = open("{0}/topo".format(reco_dir), 'w')
        write_topo(args, num_clusters, topo_file)
        topo_file.close()

        common_lib.run_kaldi_command("{cmd} {dir}/log/init_gmm.log "
                                     "gmm-init-mono {topo} {dim} "
                                     "{dir}/0.mdl {dir}/tree".format(
                                         cmd=args.cmd,
                                         topo=topo_file.name, dim=feat_dim,
                                         dir=reco_dir))
    # end if

    if args.stage <= -2:
        lex_file = open("{0}/lexicon.txt".format(reco_dir), 'w')
        for i in range(num_clusters):
            print ("{0} {0}".format(i + 1), file=lex_file)
        lex_file.close()

        logger.info("For recording {reco}, making HCLG FST".format(reco=reco))

        common_lib.run_kaldi_command(
            """{cmd} {reco_dir}/make_clg.log """
            """seq {num_clusters} \| utils/make_unigram_grammar.pl \| """
            """fstcompile \| fstdeterminizestar --use-log=true \| """
            """fstminimizeencoded \| fstpushspecial \| """
            """fstarcsort --sort_type=ilabel \| """
            """fstcomposecontext --context-size=1 --central-position=0 """
            """{lang}/ilabels_1_0 \| fstarcsort --sort_type=ilabel """
            """'>' {lang}/CLG_1_0.fst"""
            """ """.format(cmd=args.cmd, reco_dir=reco_dir,
                           num_clusters=num_clusters, lang=reco_dir))

        common_lib.run_kaldi_command(
            """{cmd} {reco_dir}/log/make_hclg.log """
            """make-h-transducer --transition-scale={tscale} """
            """{reco_dir}/ilabels_1_0 {reco_dir}/tree {reco_dir}/0.mdl \| """
            """fsttablecompose - {reco_dir}/CLG_1_0.fst \| """
            """fstdeterminizestar --use-log=true \| fstrmepslocal \| """
            """fstminimizeencoded \| """
            """add-self-loops --self-loop-scale={loopscale} """
            """--reorder=true {reco_dir}/0.mdl '>' {reco_dir}/HCLG.fst"""
            """ """.format(cmd=args.cmd, reco_dir=reco_dir,
                           tscale=args.transition_scale,
                           loopscale=args.self_loop_scale))
    # end if

    if args.stage <= -1:
        logger.info("For recording {reco}, compiling training graphs".format(
            reco=reco))

        text_file = open("{reco_dir}/text".format(reco_dir=reco_dir), 'w')
        for utt in utts:
            print ("{utt} {id}".format(utt=utt, id=cluster2int[utt2spk[utt]]),
                   file=text_file)
        text_file.close()

        common_lib.run_kaldi_command(
            "utils/split_scp.pl {reco_dir}/text {split_text}"
            "".format(reco_dir=reco_dir,
                      split_text=" ".join(
                          ["{reco_dir}/text.{x}.{nj}".format(
                              reco_dir=reco_dir, x=x, nj=args.utt_nj)
                           for x in range(1, args.utt_nj + 1)])))

        common_lib.run_job(
            """{cmd} JOB=1:{nj} {reco_dir}/log/compile_train_graphs.JOB.log """
            """compile-train-graphs {reco_dir}/tree {reco_dir}/0.mdl """
            """ "utils/make_lexicon_fst.pl {reco_dir}/lexicon.txt |"""
            """ fstcompile |" """
            """ ark,t:{reco_dir}/text.JOB.{nj} """
            """ "ark:| gzip -c > {reco_dir}/fsts.JOB.gz" """.format(
                cmd=args.cmd, nj=args.utt_nj, reco_dir=reco_dir))
    # end if

    max_gauss = args.num_gauss_per_cluster * num_clusters
    num_gauss = num_clusters
    inc_gauss = (max_gauss - num_gauss) / args.max_iter_inc

    for iter_ in range(args.num_iters):
        if args.stage > iter_:
            continue

        if iter_ <= args.max_iter_inc:
            run_align_iter(args, reco_dir, iter_, num_gauss)
        elif iter == args.max_iter_inc + 1:
            run_align_iter(args, reco_dir, iter_, 0)
        else:
            run_decode_iter(args, reco_dir, iter_, num_gauss)

        if iter_ <= args.max_iter_inc:
            num_gauss += inc_gauss

        common_lib.run_kaldi_command(
            "rm {reco_dir}/{iter}.*.acc".format(reco_dir=reco_dir,
                                                iter=iter_))
    # end for loop over iter_

    if args.stage <= args.num_iters:
        feats_rspecifier = (
            "ark,s,cs:utils/filter_scp.pl {reco_dir}/text.JOB.{nj} "
            "{data}/feats.scp | add-deltas scp:- ark:- |"
            "".format(reco_dir=reco_dir, nj=args.utt_nj, data=args.data))

        common_lib.force_symlink(
            "{0}.mdl".format(args.num_iters),
            "{0}/final.mdl".format(reco_dir))

        decode_opts = []
        decode_opts.append("--beam={0}".format(args.beam))
        decode_opts.append("--max-active={0}".format(args.max_active))
        decode_opts.append(
            "--acoustic-scale={0}".format(args.acoustic_scale))
        logger.info("Generating final segments")

        common_lib.run_job(
            """{cmd} JOB=1:{nj} {reco_dir}/log/best_path.{iter}.JOB.log """
            """gmm-decode-faster {decode_opts} """
            """ {reco_dir}/{iter}.mdl {reco_dir}/HCLG.fst """
            """ "{feats}" ark:/dev/null ark:- \| """
            """ali-to-phones --per-frame {reco_dir}/{iter}.mdl ark:- """
            """ark:- \| segmentation-init-from-ali ark:- """
            """ark:{reco_dir}/final_segmentation.JOB.ark"""
            """ """.format(cmd=args.cmd, nj=args.utt_nj,
                           reco_dir=reco_dir, iter="final",
                           decode_opts=" ".join(decode_opts),
                           feats=feats_rspecifier))
    # end if

    if args.stage <= args.num_iters + 1:
        segments_rspecifier = (
            "ark:utils/filter_scp.pl {reco_dir}/text "
            "{data}/segments | segmentation-init-from-segments "
            "--frame-overlap=0 --shift-to-zero=false - ark:- |"
            "".format(data=args.data, reco_dir=reco_dir))
        reco2utt_rspecifier = (
            """ark,t:echo '{line}' |""".format(
                line=' '.join([reco] + utts)))

        common_lib.run_kaldi_command(
            """{cmd} {reco_dir}/log/get_final_segments.log """
            """cat {reco_dir}/final_segmentation.*.ark \| """
            """segmentation-combine-segments ark:- "{segments}" """
            """ "{reco2utt}" ark:- \| """
            """segmentation-post-process --merge-adjacent-segments ark:- """
            """ ark:- \| segmentation-to-segments --frame-overlap=0 """
            """ ark:- ark,t:{reco_dir}/utt2spk {reco_dir}/segments"""
            """ """.format(cmd=args.cmd, reco_dir=reco_dir,
                           segments=segments_rspecifier,
                           reco2utt=reco2utt_rspecifier))
    # end if
    common_lib.run_kaldi_command(
        "touch {reco_dir}/.done".format(reco_dir=reco_dir))


def run(args):
    """
    We split on recordings because we have to train a HMM-GMM per recording
    with one GMM for each cluster (speaker) in the recording.
    We will also have one FST per-recording, each with its own symbols. i.e.
    the cluster-id (phone-id) for one recording has nothing to do with another
    recording.
    """

    common_lib.run_kaldi_command("utils/data/get_reco2utt.sh "
                                 "{data}".format(data=args.data))

    [stdout_val, stderr_val ] = common_lib.run_kaldi_command(
        """feat-to-dim --print-args=false """
        """ "ark:head -n 1 {data}/feats.scp | """
        """add-deltas scp:- ark:- |" -""".format(data=args.data))
    feat_dim = int(stdout_val)

    utt2spk = {}
    for line in open("{data}/utt2spk".format(data=args.data)):
        parts = line.strip().split()
        assert len(parts) == 2
        utt2spk[parts[0]] = parts[1]

    pool = multiprocessing.Pool(processes=args.num_threads)
    processes = []
    for line in open("{data}/reco2utt".format(data=args.data)):
        parts = line.strip().split()
        reco = parts[0]
        utts = parts[1:]

        logger.info("Processing recording {reco}...".format(reco=reco))
        processes.append(pool.apply_async(
            run_per_reco, args=(args, feat_dim, utt2spk, reco, utts)))
    # end for loop over reco2utt

    for p in processes:
        p.wait()
        p.get()

    common_lib.run_kaldi_command(
        "utils/copy_data_dir.sh {data} {out_data}".format(
            data=args.data, out_data=args.out_data))
    common_lib.run_kaldi_command(
        "cat {dir}/refine_*/utt2spk > {out_data}/utt2spk".format(
            dir=args.dir, out_data=args.out_data))
    common_lib.run_kaldi_command(
        "cat {dir}/refine_*/segments > {out_data}/segments".format(
            dir=args.dir, out_data=args.out_data))
    common_lib.run_kaldi_command(
        "rm {0}/{{feats.scp,cmvn.scp,spk2utt}}".format(args.out_data))
    common_lib.run_kaldi_command(
        "utils/fix_data_dir.sh {0}".format(args.out_data))



def main():
    """The main function"""
    args = get_args()
    try:
        run(args)
    except:
        logger.error("Failed re-alignment", exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
