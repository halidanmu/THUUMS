#!/bin/sh
trap "exit" INT

STATE=$1
GPU=$2

while :
do
PYTHONPATH=`pwd` THEANO_FLAGS=floatX=float32,device=$GPU python experiments/nmt/train.py --proto prototype_search_state --state $STATE
done
