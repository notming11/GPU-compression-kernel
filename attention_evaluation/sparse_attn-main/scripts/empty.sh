#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=12
#SBATCH -t 2:00:00
#SBATCH --account=def-mmehride

echo export SLURM_TMPDIR=$SLURM_TMPDIR

sleep 10000000000