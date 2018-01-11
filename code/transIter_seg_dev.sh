#!/bin/bash
for (( c = 0; c<= 50000; c = c + 1000 ))
do
   my_scripts/run_trans.sh ../data/input.dev ./result/word_uy_unphonic_em300_dev_iter_$c ../param/models/iter_$c/search_state.pkl ../param/models/iter_$c/search_model.npz 10 gpu3
done
