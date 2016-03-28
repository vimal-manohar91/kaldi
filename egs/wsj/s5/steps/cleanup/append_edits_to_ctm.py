#!/bin/env python

# Copyright 2015  Brno University of Technology (author: Karel Vesely)
#           2016  Vimal Manohar
# Apache 2.0

from __future__ import print_function
import sys, operator, argparse

# Append Levenshtein alignment of 'hypothesis' and 'reference' into 'CTM':
# (i.e. the output of 'align-text' post-processed by 'wer_per_utt_details.pl')

# The tags in the appended column are:
#  'C' = correct
#  'S' = substitution
#  'I' = insertion
#  'U' = unknown (not part of scored segment)

def GetArgs():
    parser = argparse.ArgumentParser(description =
        "Append Levenshtein alignment of 'hypothesis' and 'reference' into 'CTM':\n"
        "(i.e. the output of 'align-text' post-processed by 'wer_per_utt_details.pl')\n"
        " The tags in the appended column are:\n"
        "  'C' = correct\n"
        "  'S' = substitution\n"
        "  'I' = insertion\n"
        "  'U' = unknown (not part of scored segment)\n")

    parser.add_argument("eval_in", metavar = "<eval-in>",
                        help = "Output of 'align-text' post-processed by 'wer_per_utt_details.pl'")
    parser.add_argument("ctm_in", metavar = "<ctm-in>",
                        help = "Hypothesized CTM")
    parser.add_argument("ctm_eval_out", metavar = "<ctm-eval-out>",
                        help = "CTM appended with word-edit information")
    parser.add_argument("--special-symbol", default = "<eps>",
                        help = "Special symbol used to align insertion or deletion "
                        "in align-text binary")
    parser.add_argument("--silence-symbol",
                        help = "Must be provided to ignore silence words in the "
                        "CTM that would be present if --print-silence was true in "
                        "nbest-to-ctm binary")

    args = parser.parse_args()

    return args

def CheckArgs(args):
    if args.ctm_eval_out == "-":
        args.ctm_eval_out_handle = sys.stdout
    else:
        args.ctm_eval_out_handle = open(args.ctm_eval_out, 'w')

    if args.silence_symbol == args.special_symbol:
        print("WARNING: --silence-symbol and --special-symbol are the same", file = sys.stderr)

    return args

class CtmEvalProcessor:
    def __init__(self, args):
        self.silence_symbol = args.silence_symbol
        self.special_symbol = args.special_symbol
        self.eval_vec = dict()
        self.ctm = dict()
        self.ctm_eval = []

    def ReadEvaluation(self, eval_in):
        # Read the evalutation,
        eval_vec = self.eval_vec
        with open(eval_in, 'r') as f:
            while True:
                # Reading 4 lines encoding one utterance,
                ref = f.readline()
                hyp = f.readline()
                op = f.readline()
                csid = f.readline()
                if not ref: break
                # Parse the input,
                utt,tag,ref_vec = ref.split(' ',2)
                assert(tag == 'ref')
                utt,tag,hyp_vec = hyp.split(' ',2)
                assert(tag == 'hyp')
                utt,tag,op_vec = op.split(' ',2)
                assert(tag == 'op')
                ref_vec = ref_vec.split()
                hyp_vec = hyp_vec.split()
                op_vec = op_vec.split()
                # Fill create eval vector with symbols 'C', 'S', 'I', 'D'
                assert(utt not in eval_vec)
                eval_vec[utt] = [ (op,hyp,ref) for op,hyp,ref in zip(op_vec, hyp_vec, ref_vec) ]

    def LoadCtm(self, ctm_in):
        # Load the 'ctm' into dictionary,
        ctm = self.ctm
        with open(ctm_in) as f:
            for l in f:
                splits = l.split()
                if len(splits) == 6:
                    utt, ch, beg, dur, wrd, conf = splits
                    if not utt in ctm: ctm[utt] = []
                    ctm[utt].append((utt, ch, float(beg), float(dur), wrd, float(conf)))
                else:
                    utt, ch, beg, dur, wrd = splits
                    if not utt in ctm: ctm[utt] = []
                    ctm[utt].append((utt, ch, float(beg), float(dur), wrd))

    def ProcessInsertion(self, ctm, eval_vec, ctm_iter, eval_iter):
        ctm_appended = []
        assert(ctm_iter < len(ctm))
        assert(eval_iter < len(eval_vec))
        while (ctm[ctm_iter][4] == self.silence_symbol):
            ctm_appended.append(ctm[ctm_iter])
            ctm_iter += 1
            assert(ctm_iter < len(ctm))
        assert (ctm[ctm_iter][4] == eval_vec[eval_iter][1])
        assert (eval_vec[eval_iter][2] == self.special_symbol)
        ctm_appended.append(ctm[ctm_iter] + (self.silence_symbol, 'I'))
        ctm_iter += 1
        eval_iter += 1
        assert( [ (len(x)>=5 and x[4] == self.silence_symbol) or (len(x) >= 6) for x in ctm_appended ] )
        return ctm_appended, ctm_iter, eval_iter

    def ProcessSubstitution(self, ctm, eval_vec, ctm_iter, eval_iter):
        ctm_appended = []
        assert(ctm_iter < len(ctm))
        assert(eval_iter < len(eval_vec))
        while (ctm[ctm_iter][4] == self.silence_symbol):
            ctm_appended.append(ctm[ctm_iter])
            ctm_iter += 1
            assert(ctm_iter < len(ctm))
        assert(ctm[ctm_iter][4] == eval_vec[eval_iter][1])
        ctm_appended.append(ctm[ctm_iter] + (eval_vec[eval_iter][2],'S'))
        ctm_iter += 1
        eval_iter += 1
        assert( [ (len(x)>=5 and x[4] == self.silence_symbol) or (len(x) >= 6) for x in ctm_appended ] )
        return ctm_appended, ctm_iter, eval_iter

    def ProcessDeletion(self, ctm, eval_vec, ctm_iter, eval_iter):
        ctm_appended = []
        assert(ctm_iter <= len(ctm))
        assert(eval_iter < len(eval_vec))
        assert(eval_vec[eval_iter][1] == self.special_symbol)
        if (ctm_iter == len(ctm)):
            ctm_appended.append(ctm[ctm_iter-1][0:2] + (ctm[ctm_iter-1][2]+ctm[ctm_iter-1][3],0,self.silence_symbol,eval_vec[eval_iter][2],'D'))
        elif (ctm[ctm_iter][4] == self.silence_symbol):
            ctm_appended.append(ctm[ctm_iter] + (eval_vec[eval_iter][2],'D'))
            ctm_iter += 1
        else:
            ctm_appended.append(ctm[ctm_iter][0:3] + (0,self.silence_symbol,eval_vec[eval_iter][2],'D'))
        eval_iter += 1;
        assert( [ (len(x)>=5 and x[4] == self.silence_symbol) or (len(x) >= 6) for x in ctm_appended ] )
        return ctm_appended, ctm_iter, eval_iter

    def ProcessCorrect(self, ctm, eval_vec, ctm_iter, eval_iter):
        ctm_appended = []
        assert(ctm_iter < len(ctm))
        assert(eval_iter < len(eval_vec))
        while (ctm[ctm_iter][4] == self.silence_symbol):
            ctm_appended.append(ctm[ctm_iter])
            ctm_iter += 1
            assert(ctm_iter < len(ctm))
        assert(ctm[ctm_iter][4] == eval_vec[eval_iter][1])
        assert(ctm[ctm_iter][4] == eval_vec[eval_iter][2])
        ctm_appended.append(ctm[ctm_iter] + (eval_vec[eval_iter][2],'C'))
        ctm_iter += 1
        eval_iter += 1
        assert( [ (len(x)>=5 and x[4] == self.silence_symbol) or (len(x) >= 6) for x in ctm_appended ] )
        return ctm_appended, ctm_iter, eval_iter

    def AppendEvalToCtm(self):
        # Build the 'ctm' with 'eval' column added,
        ctm = self.ctm
        eval_vec = self.eval_vec
        for utt, utt_ctm in ctm.iteritems():
            utt_ctm.sort(key = operator.itemgetter(2)) # Sort by 'beg' time,

            utt_eval_vec = eval_vec[utt]
            # eval_vec is assumed to be in order

            ctm_iter = 0
            eval_iter = 0

            merged = []

            while eval_iter < len(utt_eval_vec):
                if utt_eval_vec[eval_iter][0] == 'I':
                    # Insertion
                    ctm_appended, ctm_iter, eval_iter = self.ProcessInsertion(utt_ctm, utt_eval_vec, ctm_iter, eval_iter)
                elif utt_eval_vec[eval_iter][0] == 'S':
                    # Substitution
                    ctm_appended, ctm_iter, eval_iter = self.ProcessSubstitution(utt_ctm, utt_eval_vec, ctm_iter, eval_iter)
                elif utt_eval_vec[eval_iter][0] == 'D':
                    # Deletion
                    ctm_appended, ctm_iter, eval_iter = self.ProcessDeletion(utt_ctm, utt_eval_vec, ctm_iter, eval_iter)
                elif utt_eval_vec[eval_iter][0] == 'C':
                    # Correct
                    ctm_appended, ctm_iter, eval_iter = self.ProcessCorrect(utt_ctm, utt_eval_vec, ctm_iter, eval_iter)
                else:
                    raise Exception('Unknown type ' + utt_eval_vec[eval_iter][0])
                merged.extend(ctm_appended)

            while ctm_iter < len(utt_ctm):
                assert(utt_ctm[ctm_iter][4] == self.silence_symbol)
                merged.append(utt_ctm[ctm_iter])
                ctm_iter += 1

            self.ctm_eval.extend(merged)

        # Sort again,
        self.ctm_eval.sort(key = operator.itemgetter(0,1,2))

    def WriteCtmEval(self, ctm_eval_out_handle):
        for tup in self.ctm_eval:
            try:
                if len(tup) == 8:
                    ctm_eval_out_handle.write('%s %s %.02f %.02f %s %f %s %s\n' % tup)
                elif len(tup) == 7:
                    ctm_eval_out_handle.write('%s %s %.02f %.02f %s %s %s\n' % tup)
                elif len(tup) == 6:
                    ctm_eval_out_handle.write('%s %s %.02f %.02f %s %s\n' % tup)
                elif len(tup) == 5:
                    ctm_eval_out_handle.write('%s %s %.02f %.02f %s\n' % tup)
                else:
                    raise Exception("Invalid line in ctm-out {0}".format(str(tup)))
            except Exception:
                raise Exception("Invalid line in ctm-out {0}".format(str(tup)))

def Main():
    args = GetArgs();
    args = CheckArgs(args)

    ctm_eval_processor = CtmEvalProcessor(args)
    ctm_eval_processor.ReadEvaluation(args.eval_in)
    ctm_eval_processor.LoadCtm(args.ctm_in)
    ctm_eval_processor.AppendEvalToCtm()
    ctm_eval_processor.WriteCtmEval(args.ctm_eval_out_handle)

if __name__ == "__main__":
    Main()
