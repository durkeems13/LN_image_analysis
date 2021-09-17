#!/bin/sh

export trainlogdir_L=$1
export trainlogdir_DC=$2
export resultsdir=$3
export rootdir_L=../../$trainlogdir_L
export rootdir_DC=../../$trainlogdir_DC
SECONDS=0

if [ ! -d "$resultsdir" ]; then
    mkdir "$resultsdir"
    mkdir "$resultsdir"/{L_inference,DC_inference,processing,analysis}
fi

cd joint_code/L_inference

echo "predict.sbatch running"
sbatch --wait predict.sbatch 
echo "predict.sbatch done"
echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo ""

cd ../DC_inference
echo "predict.sbatch running"
sbatch --wait predict.sbatch 
echo "predict.sbatch done"
echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo ""

cd ../processing
echo "combine inference pickles running"
sbatch --wait combine_inference_pkls.sbatch
echo "combine pkls done"
echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo ""


cd ../../
echo "done"
