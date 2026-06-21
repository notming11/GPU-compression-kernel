#!/bin/bash
#SBATCH --job-name=triton_bench
#SBATCH --output=benchmark_log.txt
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1     

module load StdEnv/2023
module load cuda/13.2
module load python/3.11.5

echo "Job starting..."

# 1. Activate your clean virtual environment
source /scratch/notming/triton_practice/triton_env/bin/activate

# 2. Run the Python script
python /scratch/notming/triton_practice/persistent.py

echo "Job complete!"