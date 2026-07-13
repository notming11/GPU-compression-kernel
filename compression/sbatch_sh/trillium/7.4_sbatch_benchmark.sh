#!/bin/bash
#SBATCH --job-name=7.4_benchmark        # Job name
#SBATCH --output=7.4_benchmark_%j.out    # Output file (%j = job ID)
#SBATCH --nodes=1                         # Request 1 node
#SBATCH --gpus-per-node=1                 # Request 1 GPU
#SBATCH --time=04:00:00                   # Max runtime (2hr)

# setup
module load StdEnv/2023 gcc/12.3 python/3.14.2 cuda/13.2
cd $SCRATCH/ && source .venv/bin/activate && export TRITON_HOME=$SCRATCH/.triton_cache && export TRITON_CACHE_DIR="$SCRATCH/triton_cache"&& cd compression

# run benchmark
apptainer exec --nvccli $SCRATCH/sparse.sif python 8.7_benchmark_persistent.py 7.5.1 16