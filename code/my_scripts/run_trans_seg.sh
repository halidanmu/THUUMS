#!/bin/sh
set -ex

echo $#

if [ $# -lt 5 ];
then
	echo "usage: $0 [source input] [output] [model_state] [model file] [cpu|gpu0|gpu1|..] [optional ignore_unk]"
	exit 1
fi

SOURCE=$1
OUTPUT=$2
MODEL_STATE=$3
MODEL=$4
GPU=$5
    

if [ $# -eq 5 ];
then
    PYTHONPATH=`pwd` THEANO_FLAGS=floatX=float32,device=$GPU python experiments/nmt/sample_seg.py --state $MODEL_STATE --source $SOURCE --trans $OUTPUT $MODEL
fi

if [ $# -eq 6 ];
then
    PYTHONPATH=`pwd` THEANO_FLAGS=floatX=float32,device=$GPU python experiments/nmt/sample_seg.py --state $MODEL_STATE --source $SOURCE --trans $OUTPUT --ignore-unk $MODEL
fi
