#!/bin/sh

#SBATCH -p broadwl
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --output=rgb.out
#SBATCH --error=rgb.err
#SBATCH --job-name=rgb
#export rootdir=/project2/l332/durkee_cdm3/mrcnn_human_ffpe_ss_DC/analysis/joint_preds_fold1
export READ=$rootdir/processing/final_overlays
export WRITE=$rootdir/processing/final_overlays_rgb

module load use.deprecated horovod

python make_rgb_seg_images.py --read $READ --write $WRITE
mv rgb.out $rootdir/processing
mv rgb.err $rootdir/processing
