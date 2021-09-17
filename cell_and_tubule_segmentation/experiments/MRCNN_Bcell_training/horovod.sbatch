#!/bin/bash

#SBATCH -p gpu2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4 
#SBATCH --exclusive
#SBATCH --time=36:00:00
#SBATCH --output=tp.out
#SBATCH --error=tp.err
#SBATCH --job-name=tp

export TRAIN_RUN=MP_2ch_Tcell_detection_CD3CD20_raw
export TRAIN_LOGDIR=../../train_logs/$TRAIN_RUN

if [ ! -d $TRAIN_LOGDIR ]; then
    mkdir $TRAIN_LOGDIR 
fi

module purge
module load env/rcc
module load midway2
module load use.deprecated horovod

which mpirun
which python
which nvcc

echo "SLURM_NODELIST=$SLURM_NODELIST"
echo "SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"

SERVERS=`scontrol show hostname $SLURM_NODELIST | perl -e '@a=<>;map(chomp, @a); @b = map("$_:$ENV{SLURM_NTASKS_PER_NODE}", @a); $c=join(",",@b); print $c,"\n"'`

echo "SERVERS=$SERVERS"

#mpirun -np $SLURM_NTASKS -H $SERVERS -bind-to none \
#    -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#    -mca pml ob1 -mca btl ^openib \
#    python train.py --load=../../train_logs/MP_Tcell_detection/checkpoint --logdir=../../train_logs/MP_Tcell_detection_resume1

mpirun -np $SLURM_NTASKS -H $SERVERS -bind-to none \
    -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py --logdir=$TRAIN_LOGDIR

mv tp.out $TRAIN_LOGDIR
mv tp.err $TRAIN_LOGDIR
