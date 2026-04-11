#!/bin/bash
#SBATCH --job-name=hello_nlp
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sd6701/nlp-spring-2026/HW-4/release/part-2/logs/out/%j_%x.out
#SBATCH --error=/home/sd6701/nlp-spring-2026/HW-4/release/part-2/logs/err/%j_%x.err

mkdir -p /home/sd6701/nlp-spring-2026/HW-4/release/part-2/logs/out
mkdir -p /home/sd6701/nlp-spring-2026/HW-4/release/part-2/logs/err

source /scratch/sd6701/miniconda3/etc/profile.d/conda.sh
conda activate nlp_hw4

echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "print('hello world')"
python -m pip list
