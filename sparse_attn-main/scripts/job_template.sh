#!/bin/bash
#SBATCH --ntasks-per-node=12
#SBATCH --nodes=1
#SBATCH -t 6:30:00
#SBATCH --account=def-mmehride
#SBATCH --job-name=finetune_nm

SCRIPT_TO_RUN="scripts/finetune_nm.sh"

echo "Starting SLURM job $SLURM_JOB_ID for job name $SLURM_JOB_NAME"
echo "Using SLURM temporary directory: $SLURM_TMPDIR"
echo "Received arguments: $@"


CLUSTER_NAME="$1"
shift  

USERNAME=$(whoami)

module load apptainer
export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="$SLURM_TMPDIR/data"
export HF_DATASETS_OFFLINE="1"
export HF_HUB_OFFLINE="1"
export MASTER_PORT=29501
export OMP_NUM_THREADS=12


if [ "$CLUSTER_NAME" = "fir" ]; then
    BASE_DIR="/home/${USERNAME}/projects/def-mmehride/${USERNAME}"
    DATA_DIR_TMP="${BASE_DIR}/data"
    TARBALL_PATH="${BASE_DIR}/torch-one-shot.tar"
    USE_FAKEROOT=0
elif [ "$CLUSTER_NAME" = "trillium" ]; then
    BASE_DIR="/project/rrg-mmehride/${USERNAME}"
    DATA_DIR_TMP="${BASE_DIR}/data"
    TARBALL_PATH="${BASE_DIR}/torch-one-shot.tar"
    USE_FAKEROOT=1
else
    echo "Unknown cluster name '$CLUSTER_NAME'. Expected 'fir' or 'trillium'."
    exit 1
fi


echo "Preparing container in $SLURM_TMPDIR ..."
rm -rf "$SLURM_TMPDIR/torch-one-shot.sif"
mkdir -p "$SLURM_TMPDIR/torch-one-shot.sif"

tar -xf "$TARBALL_PATH" -C "$SLURM_TMPDIR"

mkdir -p "$SLURM_TMPDIR/torch-one-shot.sif/etc/pki/tls/certs"
cp /etc/ssl/certs/ca-bundle.crt \
   "$SLURM_TMPDIR/torch-one-shot.sif/etc/pki/tls/certs/ca-bundle.crt"
echo "Executing ${SCRIPT_TO_RUN} inside Singularity..."

SINGULARITY_ARGS=(
    --bind "$PWD:/home/${USERNAME}"
    --bind "$SLURM_TMPDIR:/tmp"
    --bind "$DATA_DIR_TMP:/home/${USERNAME}/data"
    --nv "$SLURM_TMPDIR/torch-one-shot.sif"
)

if [ "$USE_FAKEROOT" -eq 1 ]; then
    singularity exec --fakeroot "${SINGULARITY_ARGS[@]}" \
        bash -lc "cd /home/$USERNAME && bash '$SCRIPT_TO_RUN' \"\$@\"" _ "$@"
else
    singularity exec "${SINGULARITY_ARGS[@]}" \
        bash "${SCRIPT_TO_RUN}" "$@"
fi

echo "Singularity execution finished successfully."
echo "SLURM Job $SLURM_JOB_ID finished."
