#!/usr/bin/env python
#-*- coding: utf8 -*-

import argparse
import cPickle
import traceback
import logging
import time
import sys

import numpy
from ctypes import *

import experiments.nmt
from experiments.nmt import\
    RNNEncoderDecoder,\
    prototype_state,\
    parse_input

from experiments.nmt.numpy_compat import argpartition
import copy

logger = logging.getLogger(__name__)

class Timer(object):

    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time

class BeamSearch(object):

    def __init__(self, enc_dec, trg_vocab, rnn_weight=1.0):
        self.enc_dec = enc_dec
        state = self.enc_dec.state
        self.eos_id = state['null_sym_target']
        self.unk_id = state['unk_sym_target']
        self.trg_vocab = trg_vocab
        self.phrase_table = {}
        self.rnn_weight = rnn_weight

    def init_lm(self, vocab_file, lm_file, ngram, weight):
        self.lm_lib = cdll.LoadLibrary('lm_score.so')
        self.lm_scorer = self.lm_lib.GetLMScore(vocab_file, lm_file, ngram)
        self.lm_lib.score.restype = c_float
        self.lm_weight = weight

    def init_tm(self, phrase_table, weights):
        self.tm_weights = weights
        for line in file(phrase_table):
            line = line.strip()
            toks = line.split('\t')
            assert len(toks) - 2 == len(weights)
            self.phrase_table[toks[0] + '\t' + toks[1]] = map(float, toks[2:])

    def lm_score(self, str):
        return self.lm_lib.score(self.lm_scorer, str)

    def trg_i2w(self, seq):
        sen = []
        for k in xrange(len(seq)):
            if seq[k] >= self.eos_id:
                sen.append('</s>')
            else:
                sen.append(self.trg_vocab[seq[k]])
        return sen

    def compile(self):
        self.comp_repr = self.enc_dec.create_representation_computer()
        self.comp_init_states = self.enc_dec.create_initializers()
        self.comp_next_probs = self.enc_dec.create_next_probs_computer()
        self.comp_next_states = self.enc_dec.create_next_states_computer()

    def get_lm_score(self, seq):
        str = ' '.join(self.trg_i2w(seq)[-5:])

        return self.lm_score(str) * self.lm_weight

    def get_tm_score(self, src_seq, aln_mat, next_word):
        assert len(src_seq) == len(aln_mat) - 1     #src_seq does not have '<eos>'
        fin_score = 0.0
        #verbose_str = '['
        for i,src_word in enumerate(src_seq):
            key = src_word + '\t' + next_word
            if key in self.phrase_table:
                tm_scores = self.phrase_table[key]
            else:
                tm_scores = [-20.0] * len(self.tm_weights)
            score = 0.0
            for s,w in zip(tm_scores, self.tm_weights):
                score += s * w

            fin_score += score * aln_mat[i]
            #verbose_str += key + ':' + str(score) + '*' + str(aln_mat[i]) + ' ||| '
        #verbose_str += ']'
        #print >> sys.stderr, verbose_str

        return fin_score

    def search(self, seqin, seq, n_samples, ignore_unk=False, minlen=1):
        src_seq = seqin.split(' ')
        c = self.comp_repr(seq)[0]
        states = map(lambda x : x[None, :], self.comp_init_states(c))
        dim = states[0].shape[1]

        num_levels = len(states)

        fin_trans = []
        fin_costs = []
        fin_aligns = []

        trans = [[]]
        costs = [0.0]
        aligns = [[]]

        for k in range(3 * len(seq)):
            if n_samples == 0:
                break

            # Compute probabilities of the next words for
            # all the elements of the beam.
            beam_size = len(trans)
            last_words = (numpy.array(map(lambda t : t[-1], trans))
                    if k > 0
                    else numpy.zeros(beam_size, dtype="int64"))

            next_probs, aln_score_mat = self.comp_next_probs(c, k, last_words, *states)
            log_probs = numpy.log(next_probs)

            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:,self.unk_id] = -numpy.inf

            if k < minlen:
                log_probs[:,self.eos_id] = -numpy.inf

            # Find the best options by calling argpartition of flatten array
            next_costs = numpy.array(costs)[:, None] - log_probs * self.rnn_weight

            flat_next_costs = next_costs.flatten()
            cands_costs_indices = argpartition(
                    flat_next_costs.flatten(),
                    n_samples * 100)[:n_samples * 100]

            # Decypher flatten indices
            voc_size = log_probs.shape[1]
            cands_trans_indices = cands_costs_indices / voc_size
            cands_word_indices = cands_costs_indices % voc_size
            cands_costs = flat_next_costs[cands_costs_indices]

            #add SMT feature scores to costs
            for i, (orig_idx, next_word, next_cost) in enumerate(
                    zip(cands_trans_indices, cands_word_indices, cands_costs)):
                lm_score = self.get_lm_score(trans[orig_idx] + [next_word])
                tm_score = self.get_tm_score(src_seq, aln_score_mat[:,orig_idx], self.trg_i2w([next_word])[0])
                cands_costs[i] += -1.0 * lm_score + -1.0 * tm_score

            best_costs_indices = argpartition(cands_costs.flatten(), n_samples)[:n_samples]
            trans_indices = cands_trans_indices[best_costs_indices]
            word_indices = cands_word_indices[best_costs_indices]
            costs = cands_costs[best_costs_indices]

            # Form a beam for the next iteration
            new_trans = [[]] * n_samples
            new_costs = numpy.zeros(n_samples)
            new_aligns = [[]] * n_samples
            new_states = [numpy.zeros((n_samples, dim), dtype="float32") for level
                    in range(num_levels)]
            inputs = numpy.zeros(n_samples, dtype="int64")
            for i, (orig_idx, next_word, next_cost) in enumerate(
                    zip(trans_indices, word_indices, costs)):
                new_trans[i] = trans[orig_idx] + [next_word]
                new_costs[i] = next_cost
                new_aligns[i] = aligns[orig_idx] + [aln_score_mat[:,orig_idx]]
                for level in range(num_levels):
                    new_states[level][i] = states[level][orig_idx]
                inputs[i] = next_word
            new_states = self.comp_next_states(c, k, inputs, *new_states)

            # Filter the sequences that end with end-of-sequence character
            trans = []
            costs = []
            aligns = []
            indices = []
            for i in range(n_samples):
                if new_trans[i][-1] != self.enc_dec.state['null_sym_target']:
                    trans.append(new_trans[i])
                    costs.append(new_costs[i])
                    aligns.append(new_aligns[i])
                    indices.append(i)
                else:
                    n_samples -= 1
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])
                    fin_aligns.append(new_aligns[i])
            states = map(lambda x : x[indices], new_states)

        if not len(fin_trans):
            if ignore_unk:
                logger.warning("Did not manage without UNK")
                return self.search(seqin, seq, n_samples, False, minlen)
            elif n_samples < 100:
                logger.warning("Still no translations: try beam size {}".format(n_samples * 2))
                return self.search(seqin, seq, n_samples * 2, False, minlen)
            else:
                logger.error("Translation failed")

        fin_trans = numpy.array(fin_trans)[numpy.argsort(fin_costs)]
        fin_aligns = numpy.array(fin_aligns)[numpy.argsort(fin_costs)]
        fin_costs = numpy.array(sorted(fin_costs))

        return fin_trans, fin_costs, fin_aligns

def indices_to_words(i2w, seq):
    sen = []
    for k in xrange(len(seq)):
        if i2w[seq[k]] == '<eol>':
            break
        sen.append(i2w[seq[k]])
    return sen

def sample(lm_model, seqin, seq, n_samples,
        sampler=None, beam_search=None,
        ignore_unk=False, normalize=False,
        alpha=1, verbose=False):
    if beam_search:
        sentences = []
        trans, costs, aligns = beam_search.search(seqin, seq, n_samples,
                ignore_unk=ignore_unk, minlen=len(seq) / 2)
        if normalize:
            counts = [len(s) for s in trans]
            costs = [co / cn for co, cn in zip(costs, counts)]
        for i in range(len(trans)):
            sen = indices_to_words(lm_model.word_indxs, trans[i])
            sentences.append(" ".join(sen))
        for i in range(len(costs)):
            if verbose:
                print "{}: {}".format(costs[i], sentences[i])
        return sentences, costs, trans, aligns
    else:
        raise Exception("I don't know what to do")


def parse_args():
    parser = argparse.ArgumentParser(
            "Sample (of find with beam-serch) translations from a translation model")
    parser.add_argument("--state",
            required=True, help="State to use")
    parser.add_argument("--beam-search",
            action="store_true", help="Beam size, turns on beam-search")
    parser.add_argument("--beam-size",
            type=int, help="Beam size")
    parser.add_argument("--alignment",
            action="store_true", help="turns on to output the alignment info")
    parser.add_argument("--nbest",
            action="store_true", help="output nbest results (with scores), turns on nbest")
    parser.add_argument("--ignore-unk",
            default=False, action="store_true",
            help="Ignore unknown words")
    parser.add_argument("--source",
            help="File of source sentences")
    parser.add_argument("--trans",
            help="File to save translations in")
    parser.add_argument("--normalize",
            action="store_true", default=False,
            help="Normalize log-prob with the word count")
    parser.add_argument("--verbose",
            action="store_true", default=False,
            help="Be verbose")
    parser.add_argument("model_path",
            help="Path to the model")
    parser.add_argument("--weights",
            required=True,help="split by comma:rnn_weight,lm_weight,tm_weight1,tm_weight2,...")
    parser.add_argument("--lm_vocab",
            required=True,help="vocab file of lm")
    parser.add_argument("--lm_file",
            required=True,help="trie file of lm")
    parser.add_argument("--pt_file",
            required=True,help="phrase table file")
    parser.add_argument("changes",
            nargs="?", default="",
            help="Changes to state")
    return parser.parse_args()

def main():
    args = parse_args()

    state = prototype_state()
    with open(args.state) as src:
        state.update(cPickle.load(src))
    state.update(eval("dict({})".format(args.changes)))
    fea_weights = map(float, args.weights.split(','))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    rng = numpy.random.RandomState(state['seed'])

    ###########################################################
    # by He Wei
    #enc_dec = RNNEncoderDecoder(state, rng, skip_init=True)
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True, compute_alignment=True)
    ###########################################################

    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)
    indx_word = cPickle.load(open(state['word_indx'],'rb'))
    idict_src = cPickle.load(open(state['indx_word'],'r'))
    trg_idx2word = cPickle.load(open(state['indx_word_target'],'r'))

    sampler = None
    beam_search = None
    if args.beam_search:
        beam_search = BeamSearch(enc_dec, trg_idx2word, fea_weights[0])
        beam_search.compile()
        beam_search.init_lm(args.lm_vocab, args.lm_file, ngram=5, weight=fea_weights[1])
        beam_search.init_tm(args.pt_file, weights=fea_weights[2:])
    else:
        sampler = enc_dec.create_sampler(many_samples=True)

    if args.source and args.trans:
        # Actually only beam search is currently supported here
        assert beam_search
        assert args.beam_size

        fsrc = open(args.source, 'r')
        ftrans = open(args.trans, 'w')

        start_time = time.time()

        n_samples = args.beam_size
        total_cost = 0.0
        logging.debug("Beam size: {}".format(n_samples))
        for i, line in enumerate(fsrc):
            seqin = line.strip()
            seq, parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src)
            if args.verbose:
                print "Parsed Input:", parsed_in
            trans, costs, _, aligns = sample(lm_model, seqin, seq, n_samples, sampler=sampler,
                    beam_search=beam_search, ignore_unk=args.ignore_unk, normalize=args.normalize)
            best = numpy.argmin(costs)
            align_str = []
            for (idx, _a) in enumerate(aligns[best]):
                align_str.append("[%s]" % ' '.join(map(str, _a)))
                #align_str.append("[%d-%d:%f,%d-%d:%f]" % (idx, _a[0], _a[1], idx, _a[2], _a[3]))

            out_str = trans[best]
            if args.alignment:
                out_str += "\t" + ' '.join(align_str)

            if args.nbest:
                nbest_trans = trans
                nbest_costs = costs
                nbest_trans = numpy.array(nbest_trans)[numpy.argsort(nbest_costs)]
                nbest_costs = numpy.array(sorted(nbest_costs))
                nbest_str = ' ||| '.join("%s | %f" % (t, c) for (t, c) in zip(nbest_trans, nbest_costs))
                out_str += "\t" + nbest_str

            print >>ftrans, out_str

            if args.verbose:
                print "[Translation]%s\t[Align]%s" % (trans[best], ' '.join(align_str))
            total_cost += costs[best]
            if (i + 1)  % 100 == 0:
                ftrans.flush()
                logger.debug("Current speed is {} per sentence".
                        format((time.time() - start_time) / (i + 1)))
        print "Total cost of the translations: {}".format(total_cost)

        fsrc.close()
        ftrans.close()
    else:
        while True:
            try:
                seqin = raw_input('Input Sequence: ')
                n_samples = int(raw_input('How many samples? '))
                alpha = None
                if not args.beam_search:
                    alpha = float(raw_input('Inverse Temperature? '))
                seq,parsed_in = parse_input(state, indx_word, seqin, idx2word=idict_src)
                print "Parsed Input:", parsed_in
            except Exception:
                print "Exception while parsing your input:"
                traceback.print_exc()
                continue

            sample(lm_model, seq, n_samples, sampler=sampler,
                    beam_search=beam_search,
                    ignore_unk=args.ignore_unk, normalize=args.normalize,
                    alpha=alpha, verbose=True)

if __name__ == "__main__":
    main()
