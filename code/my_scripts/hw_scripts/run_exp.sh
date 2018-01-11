#!/bin/sh
set -ex

if [ $# -ne 2 ];
then
	echo "usage: $0 [state] [cpu|gpu0|gpu1|...]"
	exit 1
fi

STATE=$1
GPU=$2

export LD_LIBRARY_PATH=/home/work/hewei/epd-7.1-2-rh3-x86_64/lib:$LD_LIBRARY_PATH
export PATH=/home/work/hewei/epd-7.1-2-rh3-x86_64/bin:$PATH

PYTHONPATH=`pwd` THEANO_FLAGS=floatX=float32,device=$GPU python experiments/nmt/train.py --proto prototype_search_state --state $STATE
