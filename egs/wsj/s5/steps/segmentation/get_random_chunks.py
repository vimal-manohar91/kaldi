#! /usr/bin/env python

from __future__ import print_function
import argparse
import logging
import random


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_args():
    parser = argparse.ArgumentParser(
        description="""Generate random chunks of utterances and write out a
        subsegments file.
        The subsegments file is a 'segments' file with segments
        relative to the tutterances, with lines like
        utterance_foo-1 utterance_foo 7.5 8.2
        utterance_foo-2 utterance_foo 8.9 10.1.
        The subsegments file is usable with the subsegment_data_dir.sh script.
        """)

    parser.add_argument("--min-chunk-duration", type=float, default=1,
                        help="The minimum duration of chunk (in seconds)")
    parser.add_argument("--max-chunk-duration", type=float, default=5,
                        help="""The maximum duration of chunk (in seconds).
                        If specified as 0, then there is no maximum
                        duration limit.""")
    parser.add_argument("--intersegment-duration", type=float, default=0.5,
                        help="""Offset between the end of one chunk and the
                        beginning of the next chunk.""")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the random number generator")

    parser.add_argument("in_segments_file", type=argparse.FileType('r'),
                        help="Input segments file")
    parser.add_argument("out_subsegments_file", type=argparse.FileType('w'),
                        help="Output subsegments file")

    args = parser.parse_args()

    if args.min_chunk_duration < 0.1:
        raise ValueError("min-chunk-duration must be > 0.1")

    if (args.max_chunk_duration == 0
            or args.max_chunk_duration < args.min_chunk_duration):
        raise ValueError("max-chunk-duration must be > min-chunk-duration")

    return args


def run(args):
    random.seed(args.seed)
    for line in args.in_segments_file:
        parts = line.strip().split()
        if len(parts) != 4 and len(parts) != 5:
            raise TypeError("Invalid line in segments file {0}".format(line))

        utt_id = parts[0]
        start_time = float(parts[2])
        end_time = float(parts[3])
        dur = end_time - start_time

        # Hard code the start to be somewhere close to the beginning of the
        # segment
        start = random.random() * dur * 0.1
        while start < dur - args.min_chunk_duration:
            duration = random.random() * (args.max_chunk_duration
                                          - args.min_chunk_duration)
            end = start + min(args.min_chunk_duration + duration,
                              dur - start)

            seg_id = "{0}-{1:06d}-{2:06d}".format(utt_id, int(start * 100),
                                                  int(end * 100))
            print ("{seg_id} {utt_id} {start:.3f} {end:.3f}".format(
                        seg_id=seg_id, utt_id=utt_id, start=start, end=end),
                   file=args.out_subsegments_file)
            start = end + args.intersegment_duration


def main():
    args = get_args()
    try:
        run(args)
    except:
        logger.info("Failed getting random chunk", exc_info=True)
        raise SystemExit(1)
    finally:
        try:
            for f in [args.in_segments_file, args.out_subsegments_file]:
                f.close()
        except IOError:
            raise


if __name__ == '__main__':
    main()
