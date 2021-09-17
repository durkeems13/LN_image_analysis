#!/bin/sh

#SBATCH -p broadwl 
#SBATCH --time=16:00:00
# the default partition is sandyb from Midway1 and if you want to run on
# Midway2 please 
# use broadwl or broadwl-lc partitions
#SBATCH --ntasks=280
#SBATCH --output=feat.out
#SBATCH --error=feat.err
#SBATCH --mem=2000
#SBATCH --job-name=feat

#source ~/.bashrc-new
module load parallel/latest
module load use.deprecated horovod #python/3.5.2
#cd /project2/l332/cdm3/analysis

# for a large number of tasks, the controlling node will have a large number
# of processes, so it will be necessary to change the user process limit
#ulimit -u 10000

# the --exclusive to srun make srun use distinct CPUs for each job step
# -N1 -n1 allocates a single core to each task
srun="srun --exclusive -N1 -n1"

# --delay .2 prevents overloading the controlling node
# -j is the number of tasks parallel runs so we set it to $SLURM_NTASKS
# --joblog makes parallel create a log of tasks that it has already run
# --resume makes parallel use the joblog to resume from where it has left off
# the combination of --joblog and --resume allow jobs to be resubmitted if
# necessary and continue from where they left off
parallel="parallel --delay .2 -j $SLURM_NTASKS --joblog runtask.log --resume"

# this runs the parallel command we want
# in this case, we are running a script named runtask
# parallel uses ::: to separate options. Here {1..128} is a shell expansion
# so parallel will run the runtask script for the numbers 1 through 128
# {1} is the first argument
# as an example, the first job will run like this:
# srun --exclusive -N1 -n1 ./runtask arg1:1 > runtask.1
numberindices="$(cat $rootdir/processing/indices.txt | wc -l)"
$parallel "$srun ./compute_features_parallel_batch.sh arg1:{1}" ::: {1..1020} 

mv feat.out $rootdir/processing
mv feat.err $rootdir/processing
mv runtask.log $rootdir/processing
