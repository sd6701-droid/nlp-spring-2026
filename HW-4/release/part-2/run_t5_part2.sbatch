#!/bin/bash
#SBATCH --job-name=hw4_p2_t5
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --output=/home/sd6701/nlp-spring-2026/HW-4/release/part-2/logs/out/%j_%x.out
#SBATCH --error=/home/sd6701/nlp-spring-2026/HW-4/release/part-2/logs/err/%j_%x.err
#SBATCH --export=ALL
#SBATCH --requeue

set -euo pipefail

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"
echo "========================================"

mkdir -p /home/sd6701/nlp-spring-2026/HW-4/release/part-2/logs/out
mkdir -p /home/sd6701/nlp-spring-2026/HW-4/release/part-2/logs/err

cd /home/sd6701/nlp-spring-2026/HW-4/release/part-2 || exit 1

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
CONDA_BIN="${HOME}/miniconda3/bin/conda"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable not found at ${CONDA_BIN}" >&2
  exit 1
fi

PYTHON_CMD=("${CONDA_BIN}" run --no-capture-output -n nlp_hw4 python)

RUN_ID="${SLURM_JOB_ID:-manual}"
EXPERIMENT_NAME="experiment_${RUN_ID}"
CHECKPOINT_DIR="checkpoints/ft_experiments/${EXPERIMENT_NAME}"
DEV_SQL_FILE="results/t5_ft_${EXPERIMENT_NAME}_dev.sql"
DEV_RECORDS_FILE="records/t5_ft_${EXPERIMENT_NAME}_dev.pkl"
TEST_SQL_FILE="results/t5_ft_${EXPERIMENT_NAME}_test.sql"
TEST_RECORDS_FILE="records/t5_ft_${EXPERIMENT_NAME}_test.pkl"
SUBMIT_DEV_SQL_FILE="results/t5_ft_experiment_dev.sql"
SUBMIT_DEV_RECORDS_FILE="records/t5_ft_experiment_dev.pkl"
SUBMIT_TEST_SQL_FILE="results/t5_ft_experiment_test.sql"
SUBMIT_TEST_RECORDS_FILE="records/t5_ft_experiment_test.pkl"

echo "Conda executable: ${CONDA_BIN}"
"${CONDA_BIN}" info --envs
"${PYTHON_CMD[@]}" -c "import os, sys, numpy, torch, tqdm; print('Executable:', sys.executable); print('Torch version:', torch.__version__); print('Torch path:', torch.__file__); print('CUDA available:', torch.cuda.is_available()); print('NumPy version:', numpy.__version__); print('tqdm version:', tqdm.__version__); print('Conda env:', os.environ.get('CONDA_DEFAULT_ENV', 'unset'))"
echo "Run ID: ${RUN_ID}"
echo "Experiment name: ${EXPERIMENT_NAME}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Dev SQL output: ${DEV_SQL_FILE}"
echo "Dev records output: ${DEV_RECORDS_FILE}"
echo "Test SQL output: ${TEST_SQL_FILE}"
echo "Test records output: ${TEST_RECORDS_FILE}"
echo "Canonical dev SQL: ${SUBMIT_DEV_SQL_FILE}"
echo "Canonical dev records: ${SUBMIT_DEV_RECORDS_FILE}"
echo "Canonical test SQL: ${SUBMIT_TEST_SQL_FILE}"
echo "Canonical test records: ${SUBMIT_TEST_RECORDS_FILE}"
echo "========================================"

"${PYTHON_CMD[@]}" train_t5.py \
  --finetune \
  --experiment_name "${EXPERIMENT_NAME}" \
  --max_n_epochs 10 \
  --patience_epochs 3

"${PYTHON_CMD[@]}" evaluate.py \
  --predicted_sql "${DEV_SQL_FILE}" \
  --predicted_records "${DEV_RECORDS_FILE}" \
  --development_sql data/dev.sql \
  --development_records records/ground_truth_dev.pkl

cp "${DEV_SQL_FILE}" "${SUBMIT_DEV_SQL_FILE}"
cp "${DEV_RECORDS_FILE}" "${SUBMIT_DEV_RECORDS_FILE}"
cp "${TEST_SQL_FILE}" "${SUBMIT_TEST_SQL_FILE}"
cp "${TEST_RECORDS_FILE}" "${SUBMIT_TEST_RECORDS_FILE}"

echo "Copied canonical assignment filenames:"
echo "  ${SUBMIT_DEV_SQL_FILE}"
echo "  ${SUBMIT_DEV_RECORDS_FILE}"
echo "  ${SUBMIT_TEST_SQL_FILE}"
echo "  ${SUBMIT_TEST_RECORDS_FILE}"

echo "========================================"
echo "Finished at: $(date)"
echo "========================================"
