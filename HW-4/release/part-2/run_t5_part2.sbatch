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

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"
echo "========================================"

mkdir -p /home/sd6701/nlp-spring-2026/HW-4/release/part-2/logs/out
mkdir -p /home/sd6701/nlp-spring-2026/HW-4/release/part-2/logs/err

cd /home/sd6701/nlp-spring-2026/HW-4/release/part-2 || exit 1

source /home/sd6701/miniconda3/etc/profile.d/conda.sh
conda activate nlp_hw4

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

EXPERIMENT_NAME="experiment_${SLURM_JOB_ID}"
DEV_SQL_FILE="results/t5_ft_${EXPERIMENT_NAME}_dev.sql"
DEV_RECORDS_FILE="records/t5_ft_${EXPERIMENT_NAME}_dev.pkl"
TEST_SQL_FILE="results/t5_ft_${EXPERIMENT_NAME}_test.sql"
TEST_RECORDS_FILE="records/t5_ft_${EXPERIMENT_NAME}_test.pkl"

echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "import sys, torch; print('Executable:', sys.executable); print('Torch version:', torch.__version__); print('Torch path:', torch.__file__); print('CUDA available:', torch.cuda.is_available())"
echo "Experiment name: ${EXPERIMENT_NAME}"
echo "Dev SQL output: ${DEV_SQL_FILE}"
echo "Dev records output: ${DEV_RECORDS_FILE}"
echo "Test SQL output: ${TEST_SQL_FILE}"
echo "Test records output: ${TEST_RECORDS_FILE}"
echo "========================================"

python train_t5.py \
  --finetune \
  --experiment_name "${EXPERIMENT_NAME}" \
  --max_n_epochs 10 \
  --patience_epochs 3

python evaluate.py \
  --predicted_sql "${DEV_SQL_FILE}" \
  --predicted_records "${DEV_RECORDS_FILE}" \
  --development_sql data/dev.sql \
  --development_records records/ground_truth_dev.pkl

echo "========================================"
echo "Finished at: $(date)"
echo "========================================"
