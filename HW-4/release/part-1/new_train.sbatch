#!/bin/bash
#SBATCH --job-name=hw4_p1_bert
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=/scratch/sd6701/nlp-spring-2026/HW-4/release/part-1/logs/new_train/out/%j_%x.out
#SBATCH --error=/scratch/sd6701/nlp-spring-2026/HW-4/release/part-1/logs/new_train/err/%j_%x.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

singularity exec --cleanenv --nv \
  --overlay /scratch/sd6701/overlay-25GB-500K.ext3:ro \
  /scratch/sd6701/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
  /bin/bash -lc '
set -euo pipefail

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/envs/hw4_py311/bin:/ext3/miniconda3/bin:$PATH
hash -r

cd /scratch/sd6701/nlp-spring-2026/HW-4/release/part-1

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN=/ext3/envs/hw4_py311/bin/python

NUM_EPOCHS=3
LEARNING_RATE=5e-5
BATCH_SIZE=8
RUN_ID=${SLURM_JOB_ID:-manual}

LR_TAG=${LEARNING_RATE//./p}
RUN_NAME="part1_bert_ep${NUM_EPOCHS}_lr${LR_TAG}_bs${BATCH_SIZE}_${RUN_ID}"
RUN_DIR="runs/${RUN_NAME}"
RUN_DETAILS_FILE="${RUN_DIR}/run_details.txt"
MODEL_DIR="./out"
OUTPUT_FILE="./out_original.txt"

mkdir -p "${RUN_DIR}"
cat > "${RUN_DETAILS_FILE}" <<EOF
job_id=${RUN_ID}
run_name=${RUN_NAME}
num_epochs=${NUM_EPOCHS}
learning_rate=${LEARNING_RATE}
batch_size=${BATCH_SIZE}
working_dir=$(pwd)
start_time=$(date)
EOF

echo "Python: $(${PYTHON_BIN} -c "import sys; print(sys.executable)")"
echo "Python version: $(${PYTHON_BIN} --version)"
echo "Run name: ${RUN_NAME}"
echo "Run dir: ${RUN_DIR}"
echo "Run details file: ${RUN_DETAILS_FILE}"
echo "Model dir: ${MODEL_DIR}"
echo "Output file: ${OUTPUT_FILE}"

"${PYTHON_BIN}" -c "import datasets, evaluate, numpy, torch, transformers; print(\"Torch version:\", torch.__version__); print(\"CUDA available:\", torch.cuda.is_available()); print(\"Transformers version:\", transformers.__version__); print(\"Datasets version:\", datasets.__version__); print(\"Evaluate imported:\", evaluate.__file__)"

"${PYTHON_BIN}" main.py \
  --train \
  --eval 

cp "${OUTPUT_FILE}" "${RUN_DIR}/out_original.txt"

{
  echo "end_time=$(date)"
  echo "copied_output=${RUN_DIR}/out_original.txt"
} >> "${RUN_DETAILS_FILE}"
'
