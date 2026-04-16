#!/bin/bash
#SBATCH --job-name=t5_aug
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=/scratch/sd6701/nlp-spring-2026/HW-4/release/part-2/logs/scratch_train_augmented/out/%j_%x.out
#SBATCH --error=/scratch/sd6701/nlp-spring-2026/HW-4/release/part-2/logs/scratch_train_augmented/err/%j_%x.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

singularity exec --cleanenv --nv \
  --overlay /scratch/sd6701/overlay-25GB-500K.ext3:ro \
  /scratch/sd6701/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
  /bin/bash -lc '
set -euo pipefail

export PATH=/scratch/sd6701/conda-envs/nlp_env_311/bin:/ext3/miniconda3/bin:$PATH
hash -r
PYTHON_BIN=/scratch/sd6701/conda-envs/nlp_env_311/bin/python

cd /scratch/sd6701/nlp-spring-2026/HW-4/release/part-2

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN=/scratch/sd6701/conda-envs/nlp_env_311/bin/python
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Expected python not found: ${PYTHON_BIN}"
  exit 1
fi
if ! "${PYTHON_BIN}" -c "import numpy, torch, tqdm, transformers" >/dev/null 2>&1; then
  echo "Python exists but required packages are missing from: ${PYTHON_BIN}"
  exit 1
fi
export PATH="$(dirname "${PYTHON_BIN}"):$PATH"

EPOCHS=80
PATIENCE=25
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
SCHEDULER=cosine
WARMUP_EPOCHS=3
BATCH_SIZE=32
TEST_BATCH_SIZE=16
NUM_BEAMS=8
MAX_GEN_LEN=450
MAX_GRAD_NORM=1.0
RUN_ID=${SLURM_JOB_ID:-manual}
MODEL_TAG=scr_aug
LR_TAG=${LEARNING_RATE//./p}
WD_TAG=${WEIGHT_DECAY//./p}
GRAD_TAG=${MAX_GRAD_NORM//./p}
EXPERIMENT_NAME="experiment_ec_${MODEL_TAG}_ep${EPOCHS}_pat${PATIENCE}_lr${LR_TAG}_wd${WD_TAG}_bs${BATCH_SIZE}_tbs${TEST_BATCH_SIZE}_beam${NUM_BEAMS}_gen${MAX_GEN_LEN}_sched${SCHEDULER}_warm${WARMUP_EPOCHS}_grad${GRAD_TAG}_${RUN_ID}"

RUN_DIR="runs/scr_aug_experiments/${EXPERIMENT_NAME}"
DEV_SQL_FILE="results/t5_scr_${EXPERIMENT_NAME}_dev.sql"
DEV_RECORDS_FILE="records/t5_scr_${EXPERIMENT_NAME}_dev.pkl"
TEST_SQL_FILE="results/t5_scr_${EXPERIMENT_NAME}_test.sql"
TEST_RECORDS_FILE="records/t5_scr_${EXPERIMENT_NAME}_test.pkl"
SUBMIT_TEST_SQL_FILE="results/t5_ft_experiment_ec_test.sql"
SUBMIT_TEST_RECORDS_FILE="records/t5_ft_experiment_ec_test.pkl"
RUN_DETAILS_FILE="${RUN_DIR}/run_details.txt"

mkdir -p "${RUN_DIR}"
cat > "${RUN_DETAILS_FILE}" <<EOF
job_id=${RUN_ID}
experiment_name=${EXPERIMENT_NAME}
max_n_epochs=${EPOCHS}
patience_epochs=${PATIENCE}
learning_rate=${LEARNING_RATE}
weight_decay=${WEIGHT_DECAY}
scheduler_type=${SCHEDULER}
num_warmup_epochs=${WARMUP_EPOCHS}
batch_size=${BATCH_SIZE}
test_batch_size=${TEST_BATCH_SIZE}
num_beams=${NUM_BEAMS}
max_generation_length=${MAX_GEN_LEN}
max_grad_norm=${MAX_GRAD_NORM}
train_split=train_plus_augmented
working_dir=$(pwd)
start_time=$(date)
EOF

echo "Python: $(${PYTHON_BIN} -c "import sys; print(sys.executable)")"
echo "Python version: $(${PYTHON_BIN} --version)"
echo "Experiment name: ${EXPERIMENT_NAME}"
echo "Run dir: ${RUN_DIR}"
echo "Run details file: ${RUN_DETAILS_FILE}"
echo "Dev SQL output: ${DEV_SQL_FILE}"
echo "Dev records output: ${DEV_RECORDS_FILE}"
echo "Test SQL output: ${TEST_SQL_FILE}"
echo "Test records output: ${TEST_RECORDS_FILE}"
echo "Extra-credit SQL output: ${SUBMIT_TEST_SQL_FILE}"
echo "Extra-credit records output: ${SUBMIT_TEST_RECORDS_FILE}"

"${PYTHON_BIN}" -c "import numpy, torch, tqdm, transformers; print(\"Torch version:\", torch.__version__); print(\"CUDA available:\", torch.cuda.is_available()); print(\"NumPy version:\", numpy.__version__); print(\"Transformers version:\", transformers.__version__); print(\"tqdm version:\", tqdm.__version__)"

"${PYTHON_BIN}" train_t5_scratch_augmented.py \
  --experiment_name "${EXPERIMENT_NAME}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --scheduler_type "${SCHEDULER}" \
  --num_warmup_epochs "${WARMUP_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --test_batch_size "${TEST_BATCH_SIZE}" \
  --num_beams "${NUM_BEAMS}" \
  --max_generation_length "${MAX_GEN_LEN}" \
  --max_n_epochs "${EPOCHS}" \
  --patience_epochs "${PATIENCE}" \
  --max_grad_norm "${MAX_GRAD_NORM}"

{
  echo "end_time=$(date)"
  echo "dev_sql=${DEV_SQL_FILE}"
  echo "dev_records=${DEV_RECORDS_FILE}"
  echo "test_sql=${TEST_SQL_FILE}"
  echo "test_records=${TEST_RECORDS_FILE}"
  echo "submission_sql=${SUBMIT_TEST_SQL_FILE}"
  echo "submission_records=${SUBMIT_TEST_RECORDS_FILE}"
} >> "${RUN_DETAILS_FILE}"
'
cp "${TEST_SQL_FILE}" "${SUBMIT_TEST_SQL_FILE}"
cp "${TEST_RECORDS_FILE}" "${SUBMIT_TEST_RECORDS_FILE}"
echo "Copied submission files:"
echo "  ${SUBMIT_TEST_SQL_FILE}"
echo "  ${SUBMIT_TEST_RECORDS_FILE}"