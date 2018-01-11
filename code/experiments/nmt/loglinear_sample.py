#!/usr/bin/env python
#-*- coding: utf8 -*-

import argparse
import cPickle
import traceback
import logging
import time
import sys
import math

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

    def __init__(self, enc_dec, trg_vocab, trg_vocab_reverse, vocab_reverse):
        self.enc_dec = enc_dec
        state = self.enc_dec.state
        self.eos_id = state['null_sym_target']
        self.unk_id = state['unk_sym_target']
        self.trg_vocab = trg_vocab
        self.trg_vocab_reverse = trg_vocab_reverse
        self.vocab_reverse = vocab_reverse
        self.phrase_table = {}
        self.uni_trans = {}
        #self.rnn_weight = rnn_weight

    def init_features(self, state, weights):
        #weights: 0:UNK_tm_value 1:rnn_weight 2:lm_weight 3:tm_weight 4:word_penalty_weight

        self.unk_tm_value = weights[0]
        self.weight_rnn = weights[1]
        self.weight_wp = weights[4]     #weight for word penalty

        #init lm
        print >> sys.stderr, "init language model"
        self.lm_lib = cdll.LoadLibrary('./lm_score.so')
        self.lm_scorer = self.lm_lib.GetLMScore(state['lm_vocab'], state['lm_file'], int(state['lm_ngram']))
        self.lm_lib.score.restype = c_float
        self.weight_lm = weights[2]
        self.ngram_lm = int(state['lm_ngram'])

        #init_tm
        print >> sys.stderr, "init SMT translate model"
        self.weight_tm = weights[3]
        #self.word_dict = {}
        for line in file(state['phrase_table']):
            line = line.strip()
            toks = line.split('\t')
            #assert len(toks) - 2 == len(weights)
            self.phrase_table[toks[0] + '\t' + toks[1]] = float(toks[2])
            #if toks[0] not in self.vocab_reverse:
            if toks[0] not in self.uni_trans:
                self.uni_trans[toks[0]] = []
            self.uni_trans[toks[0]].append(toks[1])


    def init_lm(self, vocab_file, lm_file, ngram, weight):
        self.lm_lib = cdll.LoadLibrary('./lm_score.so')
        self.lm_scorer = self.lm_lib.GetLMScore(vocab_file, lm_file, ngram)
        self.lm_lib.score.restype = c_float
        self.weight_lm = weight

    def init_tm(self, phrase_table, weights):
        self.tm_weights = weights
        for line in file(phrase_table):
            line = line.strip()
            toks = line.split('\t')
            assert len(toks) - 2 == len(weights)
            self.phrase_table[toks[0] + '\t' + toks[1]] = map(float, toks[2:])

    def lm_score(self, str):
        return self.lm_lib.score(self.lm_scorer, str)

    def get_lm_score(self, seq):
        if not self.lm_scorer:
            return 0.0
        str = ' '.join(seq[-self.ngram_lm:])
        score = self.lm_score(str)
        if score < -100:
            score = -10

        return score

    def get_tm_score(self, src_seq, aln_mat, next_word):
        if not self.phrase_table:
            return 0.0

        #assert len(src_seq) == len(aln_mat) - 1     #src_seq does not have '<eos>'
        fin_score = 0.0
        unk_tm_num = 0.0
        for i,src_word in enumerate(src_seq):
            key = src_word + '\t' + next_word
            if key in self.phrase_table:
                tm_score = self.phrase_table[key]
            else:
                tm_score = self.unk_tm_value
                unk_tm_num += aln_mat[i]
            fin_score += tm_score * aln_mat[i]

        return fin_score, unk_tm_num


    def get_tm_score_new(self, src_seq, aln_array, next_word):
        if not self.phrase_table:
            return 0.0

        fin_score = 0.0
        unk_tm_num = 0.0

        #if next_word == '/' or next_word == 'UNK':
        #    print "got it"
        match_idx = -1
        for k, (i, aln_score) in enumerate(aln_array):
            if i >= len(src_seq):
                continue
            src_word = src_seq[i]
            key = src_word + '\t' + next_word
            #if src_word == '8' and next_word == '800':
            #    print 'got it'
            if key in self.phrase_table:
                tm_score = self.phrase_table[key]
                if match_idx < 0:
                    match_idx = k
                #if next_word == '/':
                #    print "got it"
            elif key == 'UNK\tUNK':
                tm_score = -7
                unk_tm_num += aln_score
            else:
                tm_score = 0
                #tm_score = self.unk_tm_value
                unk_tm_num += aln_score
                #unk_count += 1
            fin_score += tm_score / aln_score
            #fin_score += tm_score * (1 - aln_score)
            #fin_score += tm_score +  math.log(aln_score)
            #fin_score += tm_score
            #fin_score += tm_score * aln_score
        if fin_score == 0 or fin_score < -25: fin_score = -25

        return fin_score, unk_tm_num, match_idx

    def trg_i2w(self, seq):
        sen = []
        for k in xrange(len(seq)):
            if seq[k] >= self.eos_id:
                sen.append('</s>')
            else:
                sen.append(self.trg_vocab[seq[k]])
        return sen

    def trg_w2i(self, seq):
        sen = []
        for k in xrange(len(seq)):
            sen.append(self.trg_vocab_reverse[seq[k]])
        return sen

    def get_tm_cand_words(self, src_seq, aln_score_mat):
        idx = numpy.argmin(aln_score_mat)
        if src_seq[idx] in self.word_dict:
            ret = self.word_dict[src_seq[idx]]
        else:
            ret = []
        return ret

    def get_uni_trans(self, src_seq):
        ret = set()
        for w in src_seq:
            if w in self.uni_trans:
                for t in self.uni_trans[w]:
                    if t not in self.trg_vocab_reverse:
                        ret.add(t)

        return ret

    def compile(self):
        self.comp_repr = self.enc_dec.create_representation_computer()
        self.comp_init_states = self.enc_dec.create_initializers()
        self.comp_next_probs = self.enc_dec.create_next_probs_computer()
        self.comp_next_states = self.enc_dec.create_next_states_computer()

    def search(self, sen, seq, n_samples, ignore_unk=False, minlen=1):
        src_seq = sen.split(' ')
        uni_trans_set = self.get_uni_trans(src_seq)
        
        #print >> sys.stderr, uni_trans_set

        c = self.comp_repr(seq)[0]
        states = map(lambda x : x[None, :], self.comp_init_states(c))
        dim = states[0].shape[1]

        num_levels = len(states)

        fin_trans = []
        fin_costs = []
        fin_str_trans = []
        #fin_infos = []

        fin_aligns = []
        fin_lm_costs = []
        fin_tm_costs = []
        fin_rnn_costs = []
        fin_unk_nums = []

        trans = [[]]
        costs = [0.0]
        str_trans = [[]]
        #infos = [[]]

        lm_costs = [[]]
        tm_costs = [[]]
        rnn_costs = [[]]
        unk_nums = [[]]
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

            next_costs = numpy.array(costs)[:, None] - log_probs * self.weight_rnn

            flat_next_costs = next_costs.flatten()
            cands_costs_indices = argpartition(
                    flat_next_costs.flatten(),
                    n_samples * 100)[:n_samples * 100]

            # Decypher flatten indices
            voc_size = log_probs.shape[1]
            cands_trans_indices = cands_costs_indices / voc_size
            cands_word_indices = cands_costs_indices % voc_size
            cands_costs = flat_next_costs[cands_costs_indices]
            cands_lm_costs = numpy.zeros(len(cands_costs))
            cands_tm_costs = numpy.zeros(len(cands_costs))
            cands_unk_nums = numpy.zeros(len(cands_costs))
            cands_rnn_costs = (-1 * log_probs).flatten()[cands_costs_indices]

            unk_trans = {}
            #add SMT feature scores to costs
            for i, (orig_idx, next_word, next_cost) in enumerate(
                    zip(cands_trans_indices, cands_word_indices, cands_costs)):

                sorted_aln_idx = numpy.argsort(-aln_score_mat[:,orig_idx])[:3]
                aln_score_array = []
                for aln_idx in sorted_aln_idx:
                    aln_score_array.append([aln_idx, aln_score_mat[aln_idx, orig_idx]])

                lm_score = -self.get_lm_score(str_trans[orig_idx] + self.trg_i2w([next_word]))
                #tm_score, unk_tm_num = self.get_tm_score(src_seq, aln_score_mat[:,orig_idx], self.trg_i2w([next_word])[0])
                tm_score, unk_tm_num, _ = self.get_tm_score_new(src_seq, aln_score_array, self.trg_i2w([next_word])[0])
                tm_score = -tm_score

                if next_word == self.unk_id:
                    #lm_score = numpy.inf
                    #tm_score = numpy.inf
                    unk_trans[orig_idx] = 'UNK'
                    _unk_score = tm_score

                    for t in uni_trans_set:
                        _ls = -self.get_lm_score(str_trans[orig_idx] + [t])
                        #_ts, _unk_tm_num = self.get_tm_score(src_seq, aln_score_mat[:,orig_idx], t)
                        _ts, _unk_tm_num, match_idx = self.get_tm_score_new(src_seq, aln_score_array, t)
                        _ts = -_ts
                        if match_idx == 0 and _ls * self.weight_lm + _ts * self.weight_tm < lm_score * self.weight_lm + tm_score * self.weight_tm:
                            lm_score = _ls
                            tm_score = _ts
                            unk_tm_num = _unk_tm_num
                            unk_trans[orig_idx] = t

                #cands_rnn_costs[i] = cands_costs[i]
                cands_costs[i] += lm_score * self.weight_lm + tm_score * self.weight_tm + self.weight_wp
                cands_lm_costs[i] = lm_score
                cands_tm_costs[i] = tm_score
                cands_unk_nums[i] = unk_tm_num

            best_costs_indices = argpartition(cands_costs.flatten(), n_samples)[:n_samples]
            trans_indices = cands_trans_indices[best_costs_indices]
            word_indices = cands_word_indices[best_costs_indices]
            costs = cands_costs[best_costs_indices]
            _lm_costs = cands_lm_costs[best_costs_indices]
            _tm_costs = cands_tm_costs[best_costs_indices]
            _unk_nums = cands_unk_nums[best_costs_indices]
            _rnn_costs = cands_rnn_costs[best_costs_indices]

            # Form a beam for the next iteration
            new_trans = [[]] * n_samples
            new_str_trans = [[]] * n_samples
            new_costs = numpy.zeros(n_samples)
            new_lm_costs = [[]] * n_samples
            new_tm_costs = [[]] * n_samples
            new_rnn_costs = [[]] * n_samples
            new_unk_nums = [[]] * n_samples
            new_aligns = [[]] * n_samples
            new_states = [numpy.zeros((n_samples, dim), dtype="float32") for level
                    in range(num_levels)]
            inputs = numpy.zeros(n_samples, dtype="int64")
            for i, (orig_idx, next_word, next_cost, _lm_score, _tm_score, _unk_num, _rnn_cost) in enumerate(
                    zip(trans_indices, word_indices, costs, _lm_costs, _tm_costs, _unk_nums, _rnn_costs)):
                new_trans[i] = trans[orig_idx] + [next_word]
                if next_word == self.unk_id:
                    new_str_trans[i] = str_trans[orig_idx] + [unk_trans[orig_idx]]
                else:
                    new_str_trans[i] = str_trans[orig_idx] + self.trg_i2w([next_word])
                new_costs[i] = next_cost
                new_lm_costs[i] = lm_costs[orig_idx] + [_lm_score]
                new_tm_costs[i] = tm_costs[orig_idx] + [_tm_score]
                new_unk_nums[i] = unk_nums[orig_idx] + [_unk_num]
                new_rnn_costs[i] = rnn_costs[orig_idx] + [_rnn_cost]
                new_aligns[i] = aligns[orig_idx] + [aln_score_mat[:,orig_idx]]
                for level in range(num_levels):
                    new_states[level][i] = states[level][orig_idx]
                inputs[i] = next_word
            new_states = self.comp_next_states(c, k, inputs, *new_states)

            # Filter the sequences that end with end-of-sequence character
            trans = []
            costs = []
            str_trans = []
            #infos = []

            lm_costs = []
            tm_costs = []
            unk_nums = []
            rnn_costs = []
            aligns = []
            indices = []
            for i in range(n_samples):
                if new_trans[i][-1] != self.enc_dec.state['null_sym_target']:
                    trans.append(new_trans[i])
                    costs.append(new_costs[i])
                    str_trans.append(new_str_trans[i])
                    #infos.append(new_infos[i])
                    lm_costs.append(new_lm_costs[i])
                    tm_costs.append(new_tm_costs[i])
                    rnn_costs.append(new_rnn_costs[i])
                    unk_nums.append(new_unk_nums[i])
                    aligns.append(new_aligns[i])
                    indices.append(i)
                else:
                    n_samples -= 1
                    fin_trans.append(new_trans[i])
                    fin_costs.append(new_costs[i])
                    fin_str_trans.append(new_str_trans[i])
                    #fin_infos.append(new_infos[i])
                    fin_lm_costs.append(new_lm_costs[i])
                    fin_tm_costs.append(new_tm_costs[i])
                    fin_rnn_costs.append(new_rnn_costs[i])
                    fin_unk_nums.append(new_unk_nums[i])
                    fin_aligns.append(new_aligns[i])

            states = map(lambda x : x[indices], new_states)

        if not len(fin_trans):
            if ignore_unk:
                logger.warning("Did not manage without UNK")
                return self.search(sen, seq, n_samples, False, minlen)
            elif n_samples < 50:
                logger.warning("Still no translations: try beam size {}".format(n_samples * 2))
                return self.search(sen, seq, n_samples * 2, False, minlen)
            elif n_samples < 100:
                logger.warning("Still no translations: try beam size {}, and --ignore UNK".format(n_samples))
                return self.search(sen, seq, n_samples, True, minlen)
            else:
                logger.err("cannot find translations, return an unreliable result")
                fin_trans = trans
                fin_str_trans = str_trans
                fin_costs = costs
                fin_lm_costs = lm_costs
                fin_tm_costs = tm_costs
                fin_rnn_costs = rnn_costs
                fin_unk_nums = unk_nums
                fin_aligns = aligns
 
            #else:
            #    logger.error("Translation failed")

        fin_trans = numpy.array(fin_trans)[numpy.argsort(fin_costs)]
        fin_aligns = numpy.array(fin_aligns)[numpy.argsort(fin_costs)]
        fin_str_trans = numpy.array(fin_str_trans)[numpy.argsort(fin_costs)]
        fin_tm_costs = numpy.array(fin_tm_costs)[numpy.argsort(fin_costs)]
        fin_lm_costs = numpy.array(fin_lm_costs)[numpy.argsort(fin_costs)]
        fin_unk_nums = numpy.array(fin_unk_nums)[numpy.argsort(fin_costs)]
        fin_rnn_costs = numpy.array(fin_rnn_costs)[numpy.argsort(fin_costs)]
        fin_costs = numpy.array(sorted(fin_costs))

        return fin_trans, fin_costs, fin_aligns, fin_lm_costs, fin_tm_costs, fin_str_trans, fin_unk_nums, fin_rnn_costs

def indices_to_words(i2w, seq):
    sen = []
    for k in xrange(len(seq)):
        if i2w[seq[k]] == '<eol>':
            break
        sen.append(i2w[seq[k]])
    return sen

def sample(lm_model, seqin, seq, n_samples, beam_search=None,
        ignore_unk=False, normalize=False, verbose=False):
    if beam_search:
        sentences = []
        trans, costs, aligns, lm_costs, tm_costs, str_trans, unk_nums, rnn_costs = beam_search.search(seqin, seq, n_samples,
                ignore_unk=ignore_unk, minlen=len(seq) / 2)
        if normalize:
            counts = [len(s) for s in trans]
            costs = [co / cn for co, cn in zip(costs, counts)]
        for i in range(len(trans)):
            #sen = indices_to_words(lm_model.word_indxs, trans[i])
            #sentences.append(" ".join(sen))
            sentences.append(" ".join(str_trans[i][:-1]))
            if verbose:
                sen = indices_to_words(lm_model.word_indxs, trans[i])
                print >>sys.stderr, " ".join(sen)
        for i in range(len(costs)):
            if verbose:
                print "{}-{}-{}: {}".format(costs[i], lm_costs, tm_costs, sentences[i])
        #unk_ids = []
        #for (i, id) in enumerate(trans):
        #    if id == beam_search.unk_id:
        #        unk_ids.append(i)

        return sentences, costs, trans, aligns, lm_costs, tm_costs, unk_nums, rnn_costs
    else:
        raise Exception("I don't know what to do")


def parse_args():
    parser = argparse.ArgumentParser(
            "Sample (of find with beam-search) translations from a translation model")
    parser.add_argument("--state",
            required=True, help="State to use")
    parser.add_argument("--beam-search",
            action="store_true", default=True, help="Beam size, turns on beam-search")
    parser.add_argument("--beam-size",
            required=True, type=int, help="Beam size")
    parser.add_argument("--alignment",
            action="store_true", help="turns on to output the alignment info")
    parser.add_argument("--nbest",
            action="store_true", help="output nbest results (with scores), turns on nbest")
    parser.add_argument("--ignore-unk",
            default=False, action="store_true",
            help="Ignore unknown words")
    parser.add_argument("--show-unk",
            default=False, action="store_true",
            help="Show unk indices")
    parser.add_argument("--source",
            required=True, help="File of source sentences")
    parser.add_argument("--trans",
            required=True, help="File to save translations in")
    parser.add_argument("--normalize",
            action="store_true", default=False,
            help="Normalize log-prob with the word count")
    parser.add_argument("--verbose",
            action="store_true", default=False,
            help="Be verbose")
    parser.add_argument("--model_path",
            required=True, help="Path to the model")
    parser.add_argument("--weights",
            help="split by comma:rnn_weight,lm_weight,tm_weight,UNK_tm_value,word_penalty_weight")
    parser.add_argument("--lm_vocab",
            help="vocab file of lm")
    parser.add_argument("--lm_file",
            help="trie file of lm")
    parser.add_argument("--pt_file",
            help="phrase table file")
    parser.add_argument("--lm_ngram",
            help="ngram language model")
    parser.add_argument("--config",
            help="config file of SMT features")
    parser.add_argument("changes",
            nargs="?", default="",
            help="Changes to state")
    return parser.parse_args()

def main():
    args = parse_args()

    state = prototype_state()
    with open(args.state) as src:
        state.update(cPickle.load(src))

    if args.config:
        state.update(eval(open(args.config).read()))

    if args.weights: state['weights'] = args.weights
    if args.lm_file: state['lm_file'] = args.lm_file
    if args.lm_vocab: state['lm_vocab'] = args.lm_vocab
    if args.pt_file: state['phrase_table'] = args.pt_file
    if args.lm_ngram: state['lm_ngram'] = args.lm_ngram

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    rng = numpy.random.RandomState(state['seed'])

    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True, compute_alignment=True)

    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)
    indx_word = cPickle.load(open(state['word_indx'],'rb'))
    idict_src = cPickle.load(open(state['indx_word'],'r'))
    trg_idx2word = cPickle.load(open(state['indx_word_target'],'r'))
    trg_word2idx = cPickle.load(open(state['word_indx_trgt'],'r'))

    #0:UNK_tm_value 1:rnn_weight 2:lm_weight 3:tm_weight 4:word_penalty_weight
    fea_weights = map(float, state['weights'].split(','))
    beam_search = BeamSearch(enc_dec, trg_idx2word, trg_word2idx, indx_word)
    beam_search.compile()
    beam_search.init_features(state, fea_weights)
    #beam_search.init_lm(state['lm_vocab'], state['lm_file'], ngram=int(state['lm_ngram']), weight=fea_weights[2])
    #beam_search.init_tm(state['phrase_table'], weights=fea_weights[3:])

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
            print >> sys.stderr, "Parsed Input:", parsed_in
        trans, costs, trans_ids, aligns, lm_costs, tm_costs, unk_nums, rnn_costs = sample(lm_model, seqin, seq, n_samples,
                beam_search=beam_search, ignore_unk=args.ignore_unk, normalize=args.normalize)
        #for (i, t) in enumerate(trans):
        #    costs[i] = costs[i] / len(t)
        best = numpy.argmin(costs)
        align_str = []
        for (idx, _a) in enumerate(aligns[best]):
            align_str.append("[%s]" % ' '.join(map(str, _a)))

        if args.nbest:
            nbest_trans = trans
            nbest_costs = costs
            nbest_lm_costs = lm_costs
            nbest_tm_costs = tm_costs
            nbest_unk_nums = unk_nums
            nbest_rnn_costs = rnn_costs
            nbest_trans = numpy.array(nbest_trans)[numpy.argsort(nbest_costs)]
            nbest_lm_costs = numpy.array(nbest_lm_costs)[numpy.argsort(nbest_costs)]
            nbest_tm_costs = numpy.array(nbest_tm_costs)[numpy.argsort(nbest_costs)]
            nbest_unk_nums = numpy.array(nbest_unk_nums)[numpy.argsort(nbest_costs)]
            nbest_rnn_costs = numpy.array(nbest_rnn_costs)[numpy.argsort(nbest_costs)]
            nbest_costs = numpy.array(sorted(nbest_costs))

            for (t, lm, tm, c, u, r) in zip(nbest_trans, nbest_lm_costs, nbest_tm_costs, nbest_costs, nbest_unk_nums, nbest_rnn_costs):
                sum_lm = numpy.sum(lm)
                sum_unk = numpy.sum(u)
                sum_tm = numpy.sum(tm)
                rnn_cost = numpy.sum(r)
                sum_wp = len(t.split(' ')) + 1
                #rnn_cost = c - sum_lm * beam_search.weight_lm - sum_tm * beam_search.weight_tm - sum_wp * beam_search.weight_wp
                pure_tm = sum_tm + sum_unk * beam_search.unk_tm_value
                #rnn_cost = sum_rnn / beam_search.weight_rnn
                #print >> ftrans, "%s ||| %f %f %f %f %f ||| 0" % (t, c, rnn_cost, sum_lm, sum_tm, sum_wp)
                #print >> ftrans, "%s ||| %f %f %f %f %f ||| 0" % (t, sum_unk * beam_search.weight_tm, -rnn_cost, -sum_lm, -pure_tm, -sum_wp)
                print >> ftrans, "%s ||| %f %f %f %f ||| 0" % (t, -rnn_cost, -sum_lm, -sum_tm, -sum_wp)
                if args.verbose:
                    print >>sys.stderr, "%s ||| %f %f %f %f %f %f %f ||| 0" % (t, sum_unk * beam_search.unk_tm_value * beam_search.weight_tm,\
                                                            -rnn_cost * beam_search.weight_rnn, \
                                                            -sum_lm * beam_search.weight_lm, \
                                                            -pure_tm * beam_search.weight_tm, \
                                                            -sum_tm * beam_search.weight_tm, \
                                                            -sum_wp * beam_search.weight_wp, c)
            print >> ftrans, ''
            #nbest_str = ' ||| '.join("%s | %f" % (t, c) for (t, c) in zip(nbest_trans, nbest_costs))
            #out_str += "\t" + nbest_str
        else:
            out_str = trans[best]
            if args.alignment:
                out_str += "\t" + ' '.join(align_str)
            if args.show_unk:
                best_ids = trans_ids[best]
                unk_ids = []
                for (i, idx) in enumerate(best_ids):
                    if idx == beam_search.unk_id:
                        unk_ids.append(i)
                out_str += "\t" + ' '.join(map(str, unk_ids))

            print >>ftrans, out_str

        if args.verbose:
            print "[Translation]%s\t[Align]%s" % (trans[best], ' '.join(align_str))
        total_cost += costs[best]
        if (i + 1)  % 100 == 0:
            ftrans.flush()
            logger.debug("Current speed is {} per sentence".
                    format((time.time() - start_time) / (i + 1)))
    print "Total cost of the translations: {}".format(total_cost)
    print "Total used time: {}".format(time.time() - start_time)

    fsrc.close()
    ftrans.close()

if __name__ == "__main__":
    main()
