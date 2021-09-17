#!/bin/sh

# this is a helper function for feat.sbatch
# sourcing environment here is probably not ideal
# but had difficulty sourcing python environment
# in a way that didn't interfere with the gnu parallel module

export INDEX=$1
echo $trainlogdir
module load use.deprecated horovod

python compute_features_parallel_batch.py --index $INDEX --trainlogdir $trainlogdir 
