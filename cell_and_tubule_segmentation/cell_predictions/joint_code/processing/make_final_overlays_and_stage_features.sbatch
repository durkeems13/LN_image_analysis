#!/bin/sh

#SBATCH -p broadwl
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --output=stage_feat.out
#SBATCH --error=stage_feat.err
#SBATCH --job-name=stage_feat

# this reserves single broadwl cpu node for stitching untiled biopsies
# writing the overlays on the images a this point is not necessary, as 
# they final processed overlays are written out later but can be good to view
# intermediate step
#export rootdir=/project2/l332/durkee_cdm3/mrcnn_human_ffpe_ss_DC/analysis/joint_DC_and_lymphocyte_preds
export PKLS_READ=$rootdir/processing/combined_inference_pkls
export IMS_READ=/project2/l332/durkee_cdm3/mrcnn_human_ffpe_ss_DC/data/MR_GT
export OVERLAYS_WRITE=$rootdir/processing/final_overlays
export PKLS_WRITE=$rootdir/processing/feature_stage_pkls

module load use.deprecated horovod

python make_final_overlays_and_stage_features.py --pkls_read $PKLS_READ --ims_read $IMS_READ --overlays_write $OVERLAYS_WRITE --pkls_write $PKLS_WRITE 

python indices_tofile.py --folder $PKLS_WRITE > $rootdir/processing/indices.txt
mv stage_feat.out $rootdir/processing/
mv stage_feat.err $rootdir/processing/
