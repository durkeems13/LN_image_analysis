#!/bin/sh

export trainlogdir=$1
export rootdir=../../$trainlogdir
SECONDS=0

cd joint_code/processing
echo "make_final_overlays_and_stage_features.sbatch running"
sbatch --wait make_final_overlays_and_stage_features.sbatch
echo "make_final_overlays_and_stage_features.sbatch done"
echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo ""

#This takes around an hour so only use if want rgb images of segmentations
echo "make_rgb_seg_images.sbatch running"
sbatch --wait make_rgb_seg_images.sbatch
echo "make_rgb_seg_images.sbatch done"
echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo ""

cd ../../
echo "done"
