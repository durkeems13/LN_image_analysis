#!/bin/bash

export TRAIN_RUN=$1
export TRAIN_LOGDIR=../../train_logs/$TRAIN_RUN

if [ ! -d $TRAIN_LOGDIR ]; then
    mkdir $TRAIN_LOGDIR 
fi

assign-gpu -n 4 python3 train.py --load=../../train_logs/Human_FFPE_Bcell_512_MD/checkpoint --logdir=$TRAIN_LOGDIR
#assign-gpu -n 4 python3 train.py --logdir=$TRAIN_LOGDIR 

