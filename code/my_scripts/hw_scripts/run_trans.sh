#!/bin/sh
set -ex

echo $#

if [ $# -lt 6 ];
then
	echo "usage: $0 [source input] [output] [model_state] [model file] [beam size] [cpu|gpu0|gpu1|..] [optional ignore_unk]"
	exit 1
fi

SOURCE=$1
OUTPUT=$2
MODEL_STATE=$3
MODEL=$4
BEAM_SIZE=$5
GPU=$6
    

export LD_LIBRARY_PATH=/home/work/hewei/epd-7.1-2-rh3-x86_64/lib:$LD_LIBRARY_PATH
export PATH=/home/work/hewei/epd-7.1-2-rh3-x86_64/bin:$PATH

if [ $# -eq 6 ];
then
    PYTHONPATH=`pwd` THEANO_FLAGS=floatX=float32,device=$GPU python experiments/nmt/sample.py --state $MODEL_STATE --beam-search --beam-size $BEAM_SIZE --source $SOURCE --trans $OUTPUT $MODEL
fi

if [ $# -eq 7 ];
then
    PYTHONPATH=`pwd` THEANO_FLAGS=floatX=float32,device=$GPU python experiments/nmt/sample.py --state $MODEL_STATE --beam-search --beam-size $BEAM_SIZE --source $SOURCE --trans $OUTPUT --ignore-unk $MODEL
fi
