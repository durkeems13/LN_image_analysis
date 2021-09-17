#!/bin/bash

module load horovod
python predict.py --index=$1 --load=$2 --predict=$3 --output_dir=$4 --tasks=$5
