#!/bin/sh

#SBATCH -p gpu2 
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --output=predict.out
#SBATCH --error=predict.err
#SBATCH --job-name=predict

# this script currently runs predictions with two gpus that should be assigned
# separately on whatever gpus are available across nodes. Can be extended to
# more or a single gpu

# currently 0 and 1 along with 2 are passed to the python script in order to
# split  the list of names of images to process
# into work for each gpu

module load use.deprecated horovod
export rootdir=/project2/l332/durkee_cdm3/mrcnn_human_ffpe_ss_DC/analysis #change after last slash--output folder
#export trainlogdir=Human_FFPE_L_fold2_resume #check this
export LOAD=../../../train_logs/$trainlogdir_L/checkpoint
export PREDICT=/project2/l332/durkee_cdm3/mrcnn_human_ffpe_ss_DC/data/MR_L_finetuning/testset #change this to your data to predict from
export OUTPUT_DIR=$rootdir/$resultsdir/L_inference/untiled_biopsy_predictions
echo "$OUTPUT_DIR"
echo ""

if [ -d "$OUTPUT_DIR" ]; then
    rm -r $OUTPUT_DIR
fi
mkdir $OUTPUT_DIR

echo "SLURM_NODELIST=$SLURM_NODELIST"
echo "SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"

SERVERS=`scontrol show hostname $SLURM_NODELIST | perl -e '@a=<>;map(chomp,@a); @b = map("$_:$ENV{SLURM_NTASKS_PER_NODE}", @a); $c=join(",",@b); print $c, "\n"'`

echo "SERVERS=$SERVERS"

#srun -N1 -n1 --gres=gpu:1 python predict.py --index=0 --load=$LOAD --predict=$PREDICT --output_dir=$OUTPUT_DIR --tasks=4 &
#srun -N1 -n1 --gres=gpu:1 python predict.py --index=1 --load=$LOAD --predict=$PREDICT --output_dir=$OUTPUT_DIR --tasks=4 &
#srun -N1 -n1 --gres=gpu:1 python predict.py --index=2 --load=$LOAD --predict=$PREDICT --output_dir=$OUTPUT_DIR --tasks=4 &
#srun -N1 -n1 --gres=gpu:1 python predict.py --index=3 --load=$LOAD --predict=$PREDICT --output_dir=$OUTPUT_DIR --tasks=4 &

mpirun -np $SLURM_NTASKS -H $SERVERS -bind-to none \
    -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml obl -mca btl ^openib \
    python predict.py --index=0 --load=$LOAD --predict=$PREDICT --output_dir=$OUTPUT_DIR --tasks=4 &
wait
mpirun -np $SLURM_NTASKS -H $SERVERS -bind-to none \
    -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml obl -mca btl ^openib \
    python predict.py --index=1 --load=$LOAD --predict=$PREDICT --output_dir=$OUTPUT_DIR --tasks=4 &
wait
mpirun -np $SLURM_NTASKS -H $SERVERS -bind-to none \
    -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml obl -mca btl ^openib \
    python predict.py --index=2 --load=$LOAD --predict=$PREDICT --output_dir=$OUTPUT_DIR --tasks=4 &
wait
mpirun -np $SLURM_NTASKS -H $SERVERS -bind-to none \
    -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml obl -mca btl ^openib \
    python predict.py --index=3 --load=$LOAD --predict=$PREDICT --output_dir=$OUTPUT_DIR --tasks=4 &

wait
#mv predict.out $rootdir/L_inference
#mv predict.err $rootdir/L_inference
