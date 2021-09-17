#!/bin/sh


export trainlogdir=$1
export rootdir=../../$trainlogdir
SECONDS=0

cd joint_code/processing # change this folder for DC v. L predictions
'''
echo "compute_features_parallel_batch.sbatch running"
sbatch --wait compute_features_parallel_batch.sbatch
echo "compute_features_parallel_batch.sbatch done"
echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo ""
'''
cd ../analysis
echo "auto_cdm_analysis.sbatch running"
sbatch --wait auto_cdm_analysis.sbatch
echo "auto_cdm_analysis.sbatch done"
echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo ""

cd ../../
echo "done"
