#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

VENV_NAME=".venv"

echo "========================================"
echo "🚀 Setting up Triton Gluon Environment"
echo "========================================"

# 1. PROTECT THE LOGIN NODE
# Redirect temporary files to your scratch space so we never hit the RAM limit
export TMPDIR=$SCRATCH/tmp
mkdir -p $TMPDIR

# Throttle compilation to 2 CPU cores so the system doesn't kill us for CPU abuse
export MAX_JOBS=2

# 2. Load necessary HPC modules
echo "==> Loading system modules..."
module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
module load cuda/13.2
module load python/3.11
echo "[✓] Modules loaded (CUDA 13.2, Python 3.11)"

# 3. Create and activate the virtual environment
if [ ! -d "$VENV_NAME" ]; then
    echo "==> Creating new Python virtual environment..."
    python3 -m venv $VENV_NAME
fi
source $VENV_NAME/bin/activate
echo "[✓] Python Virtual Environment Activated"

echo "========================================"
echo "✅ Setup Complete! "
echo "========================================"