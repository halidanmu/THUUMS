#!/bin/sh
set -ex

if [ $# -ne 4 ];
then
	echo "usage: $0 [vocab size] [source text] [target text] [output model dir]"
	exit 1
fi

VOC_SIZE=$1
SRC=$2
TRG=$3
OUT=$4

[ ! -d $OUT ] && mkdir -p $OUT

export PYTHONPATH=`pwd`
export LD_LIBRARY_PATH=/home/work/hewei/epd-7.1-2-rh3-x86_64/lib:$LD_LIBRARY_PATH
export PATH=/home/work/hewei/epd-7.1-2-rh3-x86_64/bin:$PATH

python experiments/nmt/preprocess/preprocess.py -d $OUT/vocab.src.pkl -v $VOC_SIZE -b $OUT/binarized_text.src.pkl -p $SRC
python experiments/nmt/preprocess/preprocess.py -d $OUT/vocab.trg.pkl -v $VOC_SIZE -b $OUT/binarized_text.trg.pkl -p $TRG

python experiments/nmt/preprocess/invert-dict.py $OUT/vocab.src.pkl $OUT/ivocab.src.pkl
python experiments/nmt/preprocess/invert-dict.py $OUT/vocab.trg.pkl $OUT/ivocab.trg.pkl

python experiments/nmt/preprocess/convert-pkl2hdf5.py $OUT/binarized_text.src.pkl $OUT/binarized_text.src.h5
python experiments/nmt/preprocess/convert-pkl2hdf5.py $OUT/binarized_text.trg.pkl $OUT/binarized_text.trg.h5

python experiments/nmt/preprocess/shuffle-hdf5.py $OUT/binarized_text.src.h5 $OUT/binarized_text.trg.h5 $OUT/binarized_text.src.shuf.h5 $OUT/binarized_text.trg.shuf.h5

perl my_scripts/gen_state.pl $VOC_SIZE $OUT > $OUT/my_state.py
