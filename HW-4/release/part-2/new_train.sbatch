#!/bin/bash
#SBATCH --job-name=t5_ft
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=/scratch/sd6701/nlp-spring-2026/HW-4/release/part-2/logs/new_train/out/%j_%x.out
#SBATCH --error=/scratch/sd6701/nlp-spring-2026/HW-4/release/part-2/logs/new_train/err/%j_%x.err
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

cd /scratch/sd6701/nlp-spring-2026/HW-4/release/part-2

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN=/ext3/envs/hw4_py311/bin/python

EPOCHS=40
PATIENCE=10
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
SCHEDULER=cosine
WARMUP_EPOCHS=3
BATCH_SIZE=32
TEST_BATCH_SIZE=16
NUM_BEAMS=8
MAX_GEN_LEN=450
RUN_ID=${SLURM_JOB_ID:-manual}
MODEL_TAG=ft
LR_TAG=${LEARNING_RATE//./p}
WD_TAG=${WEIGHT_DECAY//./p}
EXPERIMENT_NAME="experiment_${MODEL_TAG}_ep${EPOCHS}_pat${PATIENCE}_lr${LR_TAG}_wd${WD_TAG}_bs${BATCH_SIZE}_tbs${TEST_BATCH_SIZE}_beam${NUM_BEAMS}_gen${MAX_GEN_LEN}_sched${SCHEDULER}_warm${WARMUP_EPOCHS}_${RUN_ID}"

RUN_DIR="runs/ft_experiments/${EXPERIMENT_NAME}"
DEV_SQL_FILE="results/t5_ft_${EXPERIMENT_NAME}_dev.sql"
DEV_RECORDS_FILE="records/t5_ft_${EXPERIMENT_NAME}_dev.pkl"
TEST_SQL_FILE="results/t5_ft_${EXPERIMENT_NAME}_test.sql"
TEST_RECORDS_FILE="records/t5_ft_${EXPERIMENT_NAME}_test.pkl"
SUBMIT_DEV_SQL_FILE="results/t5_ft_experiment_dev.sql"
SUBMIT_DEV_RECORDS_FILE="records/t5_ft_experiment_dev.pkl"
SUBMIT_TEST_SQL_FILE="results/t5_ft_experiment_test.sql"
SUBMIT_TEST_RECORDS_FILE="records/t5_ft_experiment_test.pkl"
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

"${PYTHON_BIN}" -c "import numpy, torch, tqdm; print(\"Torch version:\", torch.__version__); print(\"CUDA available:\", torch.cuda.is_available()); print(\"NumPy version:\", numpy.__version__); print(\"tqdm version:\", tqdm.__version__)"

"${PYTHON_BIN}" train_t5.py \
  --finetune \
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
  --patience_epochs "${PATIENCE}"

"${PYTHON_BIN}" evaluate.py \
  --predicted_sql "${DEV_SQL_FILE}" \
  --predicted_records "${DEV_RECORDS_FILE}" \
  --development_sql data/dev.sql \
  --development_records records/ground_truth_dev.pkl

cp "${DEV_SQL_FILE}" "${SUBMIT_DEV_SQL_FILE}"
cp "${DEV_RECORDS_FILE}" "${SUBMIT_DEV_RECORDS_FILE}"
cp "${TEST_SQL_FILE}" "${SUBMIT_TEST_SQL_FILE}"
cp "${TEST_RECORDS_FILE}" "${SUBMIT_TEST_RECORDS_FILE}"

{
  echo "end_time=$(date)"
  echo "canonical_dev_sql=${SUBMIT_DEV_SQL_FILE}"
  echo "canonical_dev_records=${SUBMIT_DEV_RECORDS_FILE}"
  echo "canonical_test_sql=${SUBMIT_TEST_SQL_FILE}"
  echo "canonical_test_records=${SUBMIT_TEST_RECORDS_FILE}"
} >> "${RUN_DETAILS_FILE}"
'
