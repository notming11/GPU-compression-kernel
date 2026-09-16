#!/bin/bash


# cluster 
CLUSTER=trillium #fir

# Model family 
MODEL_NAME=llama3.2

# Optimizer & LR
OPTIMIZERS=("adamw_torch")
LEARNING_RATES=("1e-3" "1e-2")   


LORA_RANKS=(4 8 16)
LORA_QR_S=1  
LR_SCALER_BS=(1 8)
WEIGHT_DECAYS=(0.01)
GLOBAL_BATCH_SIZES=(128 256 512)
LORA_INITS=("qr" "random")
NMS=("1:2" "2:4")
SLIDING_WINDOWS=(0 128 256 512)
TRAIN_STEPS=1000
PRUNED_MATRIX=("q" "k" "v")
# training
LOCAL_BATCH_SIZE_ADAM=4
SEQ_LEN=1024

# slurm
NGPUS_PER_NODE=4
NTASKS_PER_NODE=$((12 * NGPUS_PER_NODE))
MEM=$((64 * NGPUS_PER_NODE))

SLURM_SCRIPT="scripts/job_template.sh"

if [ "$MODEL_NAME" == "llama2" ]; then
    MODEL_PREFIX="meta-llama/Llama-2-"
    MODEL_POSTFIX="-hf"
    MODEL_SIZE_LIST="7b"
elif [ "$MODEL_NAME" == "opt" ]; then
    MODEL_PREFIX="facebook/opt-"
    MODEL_POSTFIX=""
    MODEL_SIZE_LIST="125m"
elif [ "$MODEL_NAME" == "llama3.2" ]; then
    MODEL_PREFIX="meta-llama/Llama-3.2-"
    MODEL_POSTFIX=""
    MODEL_SIZE_LIST="1B"
elif [ "$MODEL_NAME" == "llama3.1" ]; then
    MODEL_PREFIX="meta-llama/Llama-3.1-"
    MODEL_POSTFIX=""
    MODEL_SIZE_LIST="8B"
elif [ "$MODEL_NAME" == "gemma3" ]; then
    MODEL_PREFIX="google/gemma-3-"
    MODEL_POSTFIX="-pt"
    MODEL_SIZE_LIST="1b"
elif [ "$MODEL_NAME" == "qwen2.5" ]; then
    MODEL_PREFIX="Qwen/Qwen2.5-"
    MODEL_POSTFIX=""
    MODEL_SIZE_LIST="0.5B"
elif [ "$MODEL_NAME" == "smollm2" ]; then
    MODEL_PREFIX="HuggingFaceTB/SmolLM2-"
    MODEL_POSTFIX=""
    MODEL_SIZE_LIST="360M"
else
    echo "Unknown MODEL_NAME: $MODEL_NAME"
    exit 1
fi

echo "Starting job submission loop..."

job_count=0

# --- Sweeps ----------------------------------------------------------------

for MODEL_SIZE in $MODEL_SIZE_LIST; do
  FULL_MODEL="${MODEL_PREFIX}${MODEL_SIZE}${MODEL_POSTFIX}"

  for p_m in "${PRUNED_MATRIX[@]}"; do
  for opt in "${OPTIMIZERS[@]}"; do
    local_bs=$LOCAL_BATCH_SIZE_ADAM

    for lr in "${LEARNING_RATES[@]}"; do
      for wd in "${WEIGHT_DECAYS[@]}"; do
        for global_bs in "${GLOBAL_BATCH_SIZES[@]}"; do
          for lora_rank in "${LORA_RANKS[@]}"; do
            lora_alpha="$lora_rank"   # alpha = rank
            for lr_scaler_B in "${LR_SCALER_BS[@]}"; do
              for lora_init in "${LORA_INITS[@]}"; do
                for nm in "${NMS[@]}"; do
                  for sw in "${SLIDING_WINDOWS[@]}"; do

                      JOB_NAME="ft_${MODEL_NAME}_${MODEL_SIZE}_pm_${p_m}_lr${lr}_wd${wd}_opt${opt}_gbs${global_bs}_r${lora_rank}_lrB${lr_scaler_B}_init${lora_init}_nm$(echo "$nm" | tr ':' '-')_sw${sw}"
                    JOB_NAME=$(echo "$JOB_NAME" | sed 's/e-/em/' | sed 's/[^A-Za-z0-9._-]/_/g')

                    SAVE_MODEL_PATH="saved_models/${JOB_NAME}.pt"

                    echo "--------------------------------------------------"
                    echo "Submitting job #$((job_count + 1)): ${JOB_NAME}"
                    echo "  Model           : ${FULL_MODEL}"
                    echo "  LR              : ${lr}"
                    echo "  WD              : ${wd}"
                    echo "  Optimizer       : ${opt}"
                    echo "  Global BS       : ${global_bs}"
                    echo "  Local BS        : ${local_bs}"
                    echo "  LoRA rank/alpha : ${lora_rank}/${lora_alpha}"
                    echo "  lr_scaler_B     : ${lr_scaler_B}"
                    echo "  LoRA init       : ${lora_init}"
                    echo "  n:m             : ${nm}"
                    echo "  Sliding window  : ${sw}"
                    echo "  Seq len         : ${SEQ_LEN}"
                    echo "  Save path       : ${SAVE_MODEL_PATH}"

                      sbatch --account=rrg-mmehride \
                      --job-name="${JOB_NAME}" \
                      --gpus-per-node=${NGPUS_PER_NODE} \
                      --ntasks-per-node=${NTASKS_PER_NODE} \
                      "${SLURM_SCRIPT}" \
                        "${CLUSTER}" \
                        --model "${FULL_MODEL}" \
                        --save_model_path "${SAVE_MODEL_PATH}" \
                        --optimizer "${opt}" \
                        --lr "${lr}" \
                        --lr_scaler_B "${lr_scaler_B}" \
                        --wd "${wd}" \
                        --train_steps "${TRAIN_STEPS}" \
                        --global_bs "${global_bs}" \
                        --local_bs "${local_bs}" \
                        --lora_rank "${lora_rank}" \
                        --lora_alpha "${lora_alpha}" \
                        --lora_init "${lora_init}" \
                        --lora_qr_s "${LORA_QR_S}" \
                        --nm "${nm}" \
                        --sliding_window "${sw}" \
                          --seq_len "${SEQ_LEN}" \
                          --pruned_matrix "${p_m}"

                    ((job_count++))
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "--------------------------------------------------"
echo "Finished submitting ${job_count} jobs."
