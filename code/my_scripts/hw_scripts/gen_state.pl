#!/usr/bin/perl
use strict;
use warnings;

die "$0 [vocab size] [model_path] > [output]" if @ARGV != 2;

my $vocab_size = $ARGV[0];
my $path = $ARGV[1];

print "{\n";
print "'target':['$path/binarized_text.trg.h5'],\n";
print "'source':['$path/binarized_text.src.h5'],\n";
print "'indx_word':'$path/ivocab.src.pkl',\n";
print "'indx_word_target':'$path/ivocab.trg.pkl',\n";
print "'word_indx':'$path/vocab.src.pkl',\n";
print "'word_indx_trgt':'$path/vocab.trg.pkl',\n";
print "'null_sym_source':$vocab_size,\n";
print "'null_sym_target':$vocab_size,\n";
my $n_sym = $vocab_size + 1;
print "'n_sym_source':$n_sym,\n";
print "'n_sym_target':$n_sym,\n";
print "'seqlen':50,\n";
print "'bs':80,\n";
print "'dim':1000,\n";
print "'rank_n_approx':620,\n";
print "'prefix':'search_',\n";
print "'reload':False,\n";
print "'hookFreq':500,\n";
print "'copy_model_freq':10000,\n";
print "'copy_model_path':'$path/models',\n";

print "}\n";
