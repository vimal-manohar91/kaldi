#!/bin/env python

# Copyright 2015  Brno University of Technology (author: Karel Vesely)
#           2016  Vimal Manohar
# Apache 2.0

from __future__ import print_function
import sys, operator, argparse

def GetArgs():
    parser = argparse.ArgumentParser(description =
        "Convert the CTM file appended with information about "
        "Levenshtein alignment information of 'hypothesis' and 'reference' "
        "into a pair of segments and text files\n")

    parser.add_argument("ctm_eval_file", metavar = "<ctm-eval-in>",
                        help = "CTM file appended with Levenshtein "
                        "edit information")
    parser.add_argument("segments", metavar = "<segments-out>",
                        help = "Write output utterance-level segments")
    parser.add_argument("text", metavar = "<text-out>",
                        help = "Write text file corresponding to the "
                        "utterance-level segments")
    parser.add_argument("--word-list", help = "List of words in vocabulary")
    parser.add_argument("--silence-symbol", default = "<eps>",
                        help = "Must be provided to ignore silence words in the "
                        "CTM that would be present if --print-silence was true in "
                        "nbest-to-ctm binary")
    parser.add_argument("--frame-shift", type = float, default = 10,
                        help = "Frame shift in milliseconds")
    parser.add_argument("--max-segment-length", type = int, default = 1000,
                        help = "Maximum length of segment allowed in number of frames")
    parser.add_argument("--min-segment-length", type = int, default = 150,
                        help = "Soft threshold on minimum length of segment allowed in number of frames")
    parser.add_argument("--overlap-length", type = int, default = 50,
                        help = "Number of frames of overlap when splitting long segments into pieces")
    parser.add_argument("--pad-length", type = int, default = 10,
                        help = "Pad boundaries of segments by this frames")
    parser.add_argument("--oov-symbol", default = "<UNK>",
                        help = "Symbol of OOV words");

    args = parser.parse_args()

    return args


def CheckArgs(args):
    args.segments_handle = open(args.segments, 'w')
    args.text_handle = open(args.text, 'w')

    assert(args.frame_shift > 1)
    assert(args.max_segment_length > 0)
    assert(args.min_segment_length > 0)
    assert(args.overlap_length > 0)

    return args

class Segmenter:
    def __init__(self, args):
        self.silence_symbol = args.silence_symbol
        self.max_segment_length = args.max_segment_length
        self.min_segment_length = args.min_segment_length
        self.overlap_length = args.overlap_length
        self.frame_shift = args.frame_shift
        self.word_list = args.word_list
        self.pad_length = args.pad_length
        self.oov_symbol = args.oov_symbol
        self.ctm_eval = dict()
        self.method = 1

    def LoadCtmEval(self, ctm_eval_file):
        for line in open(ctm_eval_file):
            splits = line.strip().split()
            splits[2] = int(float(splits[2]) / self.frame_shift * 1000)
            splits[3] = int(float(splits[3]) / self.frame_shift * 1000 + 0.5)

            if splits[0] not in self.ctm_eval:
                self.ctm_eval[splits[0]] = []
            self.ctm_eval[splits[0]].append(splits)

    def ProcessCtmEvalForUtt(self, utt):
        ctm_eval = self.ctm_eval[utt]
        ctm_eval.sort(key = lambda x:(x[2],x[2]+x[3])) # Sort by 'beg' time,

        word_goodness = []

        for i in range(0, len(ctm_eval)):
            word_goodness.append(self.GetWordGoodness(ctm_eval, i))


        assert(len(word_goodness) == len(ctm_eval))
        return word_goodness

    def GetFrameLevelEdits(self, ctm_eval, i):
        if ctm_eval[i][4] != self.silence_symbol:
            if ctm_eval[i][-2] == self.special_symbol:
                edit_type = EditType.C
            else:
                edit_type = GetEditType(ctm_eval[i][-1])
        else:
            if ctm_eval[i][-1] == 'D':
                edit_type = EditType.XD
            else:
                edit_type = 'X'

        return [ edit_type for i in range(0,ctm_eval[i][3]) ]

    def GetWordGoodness(self, ctm_eval, i):
        if ctm_eval[i][-1] == 'C':
            return 1

        if ctm_eval[i][4] == self.silence_symbol:
            if ctm_eval[i][-1] == 'D':
                # if current word is silence, which is a deletion
                return 0
            assert(len(ctm_eval[i]) <= 6 or ctm_eval[-1] == 'C')

            if ( (i+1 < len(ctm_eval) and ctm_eval[i+1][-1] != 'C')
                    or (i > 0 and ctm_eval[i-1][-1] != 'C') ):
                # adjacent word is not correct
                return 2

            # current word is silence which is correct
            return 1

        if self.word_list is not None and ctm_eval[i][-2] not in self.word_list:
            # current reference word is oov
            if ( (i+1 < len(ctm_eval) and ctm_eval[i+1][-1] != 'C')
                    or (i > 0 and ctm_eval[i-1][-1] != 'C') ):
                # adjacent word is not correct
                return 0
            return 1

        # current word is neither correct nor silence and reference is not oov
        return 0

    def GetGoodnessForUtt(self, utt, word_goodness):
        ctm_eval = self.ctm_eval[utt]
        assert(len(ctm_eval) == len(word_goodness))

        N = sum([x[3] for x in ctm_eval ])
        goodness = [ 0 for x in range(0,N) ]

        for i in range(0, len(ctm_eval)):
            beg = ctm_eval[i][2]
            end = beg + ctm_eval[i][3]
            goodness[beg:end] = [ word_goodness[i] for x in range(beg, end) ]
        return goodness

    def GoodnessToSegmentsMethod1(self, goodness):
        segments = []
        i = 0
        start = -1
        while i < len(goodness):
            if i == 0 and goodness[i] == 1:
                start = 0
            if start < 0 and goodness[i] == 1:
                start = i
            if start >= 0 and goodness[i] != 1:
                length = i - start
                pad_end = i
                while goodness[pad_end] == 2 and length < self.max_segment_length:
                    pad_end += 1
                    length += 1

                length = i - start
                pad_start = start - 1
                while goodness[pad_start] == 2 and length < self.max_segment_length:
                    pad_start -= 1
                    length += 1
                pad_start += 1

                if pad_start < start and pad_end > i:
                    start -= int((start - pad_start) / 2)
                    i += int((pad_end - i) / 2)
                elif pad_start < start:
                    start = pad_start
                elif pad_end > i:
                    i = pad_end

                segments.append((start,i))
                start = -1
            i += 1
        if start >= 0:
            segments.append((start, len(goodness)))
        return segments

    def GoodnessToSegments(self, goodness):
        if self.method == 1:
            return self.GoodnessToSegmentsMethod1(goodness)

    def FrameLevelEditsToSegments(self, edits):
        segments = []
        i = 0
        start = -1
        while i < len(goodness):
            if i == 0 and edits[i] == 1:
                start = 0
            if start < 0 and goodness[i] == 1:
                start = i
            if start >= 0 and goodness[i] != 1:
                length = i - start
                pad_end = i
                while goodness[pad_end] == 2 and length < self.max_segment_length:
                    pad_end += 1
                    length += 1

                length = i - start
                pad_start = start - 1
                while goodness[pad_start] == 2 and length < self.max_segment_length:
                    pad_start -= 1
                    length += 1
                pad_start += 1

                if pad_start < start and pad_end > i:
                    start -= int((start - pad_start) / 2)
                    i += int((pad_end - i) / 2)
                elif pad_start < start:
                    start = pad_start
                elif pad_end > i:
                    i = pad_end

                segments.append((start,i))
                start = -1
            i += 1
        if start >= 0:
            segments.append((start, len(goodness)))
        return segments

    def SplitLongSegments(self, segments):
        i = 0
        num_segments = len(segments)
        while i < num_segments:
            start, end = segments[i]
            if end - start > self.max_segment_length + self.overlap_length:
                segments.insert(i, (start, start + self.max_segment_length))
                num_segments += 1
                i += 1
                segments[i] = (start + self.max_segment_length - self.overlap_length, end)
            else:
                i += 1
        assert(len(segments) == num_segments)

    def ExtractTextForSegments(self, utt, segments):
        ctm_eval = self.ctm_eval[utt]
        ctm_eval.sort(key = lambda x:(x[2],x[2]+x[3])) # Sort by 'beg' time,
        segments.sort(key = lambda x:(x[0],x[1]))

        texts = []
        for k, tup in enumerate(segments):
            beg, end = tup
            beg = max(beg - self.pad_length, 0)
            if k < len(segments):
                end += self.pad_length
            text = []
            i = next((i for i,x in enumerate(ctm_eval) if x[2] + x[3] >= beg), len(ctm_eval))
            j = next((j for j,x in enumerate(ctm_eval) if x[2] + x[3] > end), len(ctm_eval))
            if (i == len(ctm_eval)):
                continue
            for n in range(i,j):
                if n == i:
                    if (ctm_eval[n][4] != self.silence_symbol):
                        text.append(self.oov_symbol)
                else:
                    if (ctm_eval[n][4] != self.silence_symbol):
                        text.append(ctm_eval[n][-2])
            if j < len(ctm_eval) and (ctm_eval[j][4] != self.silence_symbol):
                text.append(self.oov_symbol)
            texts.append((beg,end,text))
        return texts

    def WriteSegmentsAndText(self, utt, texts, segments_handle, text_handle):
        for beg, end, text_tup in texts:
            if len(text_tup) == 0 or all([x == self.oov_symbol for x in text_tup]):
                continue
            text = ' '.join(text_tup)
            utt_id = '%s-%06d-%06d' % (utt,beg,end)
            print ('{utt} {reco} {beg} {end}'.format(utt = utt_id, reco = utt,
                beg = float(beg) * self.frame_shift / 1000.0,
                end = float(end) * self.frame_shift / 1000.0),
                file = segments_handle)
            print ('{utt} {text}'.format(utt = utt_id, text = text),
                    file = text_handle)

def Main():
    args = GetArgs()
    args = CheckArgs(args)

    segmenter = Segmenter(args)
    segmenter.LoadCtmEval(args.ctm_eval_file)

    for utt in segmenter.ctm_eval:
        word_goodness = segmenter.ProcessCtmEvalForUtt(utt)
        goodness = segmenter.GetGoodnessForUtt(utt, word_goodness)
        segments = segmenter.GoodnessToSegments(goodness)
        segmenter.SplitLongSegments(segments)
        texts = segmenter.ExtractTextForSegments(utt, segments)
        segmenter.WriteSegmentsAndText(utt, texts, args.segments_handle, args.text_handle)

if __name__ == "__main__":
    Main()
