#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys

import numpy

import experiments.nmt
from experiments.nmt import RNNEncoderDecoder, prototype_state

logger = logging.getLogger(__name__)
def parsed_sentence(line, word2idx, state, num_sym, unk_sym = -1, null_sym = -1, raise_unk = True):
    input = []
    if unk_sym < 0:
        unk_sym = state['unk_sym_source']
    if null_sym < 0:
        null_sym = state['null_sym_source']
    seqin = line.split()
    seqlen = len(seqin)
    seq = numpy.zeros(seqlen+1, dtype='int64')
    for idx,sx in enumerate(seqin):
        seq[idx] = word2idx.get(sx, unk_sym)
        if seq[idx] >= num_sym: #n_sym_source
            seq[idx] = unk_sym
        if seq[idx] == unk_sym and raise_unk:
            print "Unknown word {}".format(sx)
            #print seqin
        if seq[idx] == unk_sym:
            input.append('U'+sx)
        else:
            input.append(sx)
    seq[-1] = null_sym
    return seq, input

def comput_alignment(source_file, target_file, output_file, alignment_fun, word_indx_src, word_indx_trg, state):
    fw = open(output_file, 'w')
    cnt = 0
    fws = open(output_file + '.align.src', 'w')
    fwt = open(output_file + '.align.trg', 'w')
    for src, trg in zip(open(source_file, 'r').readlines(), open(target_file, 'r').readlines()):
        src_seq, input_src = parsed_sentence(src, word_indx_src, state, state['n_sym_source'], state['unk_sym_source'], state['null_sym_source'])
        trg_seq, input_trg = parsed_sentence(trg, word_indx_trg, state, state['n_sym_target'], state['unk_sym_target'], state['null_sym_target'])
        fws.write(' '.join(input_src) + '\n')
        fwt.write(' '.join(input_trg) + '\n')
        probs, alignment = alignment_fun(src_seq, trg_seq)
        #(seq_len, batch_size, dim)
        fw.write("="*10 + str(cnt) + "="*10 + "\n")
        print alignment.shape
        for i in range(alignment.shape[0]):
            s = ""
            for j in range(alignment.shape[1]):
                s += str(alignment[i][j][0])+" "
            fw.write(s.strip() + '\n')
        fw.flush()
        cnt += 1
    fw.close()
    fws.close()
    fwt.close()

def parse_args():
    parser = argparse.ArgumentParser(
            "Fina alignment matrix between parallel sentences")
    parser.add_argument("--state",
            required=True, help="State to use")
    parser.add_argument("--source", required = True,
            help="File of source sentences")
    parser.add_argument("--target", required = True, help = "File of target sentences")
    parser.add_argument("--output", default = "alignment.out",
            help="File to save alignment information in")
    parser.add_argument("model_path", help="Path to the model")
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

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    rng = numpy.random.RandomState(state['seed'])

    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True, compute_alignment=True)

    enc_dec.build()
    lm_model = enc_dec.create_lm_model()
    lm_model.load(args.model_path)

    alignment_fun = enc_dec.create_probs_computer(return_alignment = True)



    word_indx_src = cPickle.load(open(state['word_indx'],'rb'))
    word_indx_trg = cPickle.load(open(state['word_indx_trgt'], 'rb'))
    source_file = args.source
    target_file = args.target
    output_file = args.output

    comput_alignment(source_file, target_file, output_file, alignment_fun, word_indx_src, word_indx_trg, state)

if __name__ == "__main__":
    main()
