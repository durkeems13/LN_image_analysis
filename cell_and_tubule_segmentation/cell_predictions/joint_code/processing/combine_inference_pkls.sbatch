#!/bin/sh

#SBATCH -p broadwl
#SBATCH --time=25:30:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --output=combine.out
#SBATCH --error=combine.err
#SBATCH --job-name=stage_feat

# this reserves single broadwl cpu node for stitching untiled biopsies
# writing the overlays on the images a this point is not necessary, as 
# they final processed overlays are written out later but can be good to view
# intermediate step
export rootdir=/project2/l332/durkee_cdm3/mrcnn_human_ffpe_ss_DC/analysis
export resultsdir=MRhybrid_on_MRtestset
export L_PKLS_READ=$rootdir/$resultsdir/L_inference/untiled_biopsy_predictions
export DC_PKLS_READ=$rootdir/$resultsdir/DC_inference/untiled_biopsy_predictions
export PKLS_WRITE=$rootdir/$resultsdir/processing/combined_inference_pkls

module load use.deprecated horovod

python combine_inference_pkls.py --l_pkls_read $L_PKLS_READ --dc_pkls_read $DC_PKLS_READ --pkls_write $PKLS_WRITE 

#mv combine.out $rootdir/processing/
#mv combine.err $rootdir/processing/
