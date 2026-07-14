#!/bin/bash
#SBATCH --job-name=7.3_max_shape_benchmark        # Job name
#SBATCH --output=7.3_max_shape_benchmark_%j.out    # Output file (%j = job ID)
#SBATCH --nodes=1                         # Request 1 node
#SBATCH --gpus-per-node=h100:1                 # Request 1 GPU
#SBATCH --time=02:00:00                   # Max runtime (2hr)
#SBATCH --mem=188G

# setup
module load StdEnv/2023 gcc/12.3 python/3.14.2 cuda/13.2 apptainer/1.4.5
cd $SCRATCH/ && source .venv/bin/activate && export TRITON_HOME=$SCRATCH/.triton_cache && export TRITON_CACHE_DIR="$SCRATCH/triton_cache"&& cd GPU-compression-kernel/compression
nvidia-smi
apptainer exec --nv $SCRATCH/sparse.sif python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count()); torch.cuda.init()"
# run benchmark
apptainer exec --nv $SCRATCH/sparse.sif python 8.7_benchmark_persistent.py 7.3 16