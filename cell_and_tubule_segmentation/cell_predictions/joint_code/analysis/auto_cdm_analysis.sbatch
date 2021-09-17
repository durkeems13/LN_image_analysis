#!/bin/sh

#SBATCH -p broadwl 
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --output=analysis.out
#SBATCH --error=analysis.err
#SBATCH --job-name=analysis

module load use.deprecated
module load horovod

printenv

python auto_cdm_analysis.py --trainlogdir $trainlogdir
mv analysis.err ../../$trainlogdir/analysis
mv analysis.out ../../$trainlogdir/analysis
