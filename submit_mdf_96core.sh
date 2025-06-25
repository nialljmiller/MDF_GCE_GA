#!/bin/bash
#SBATCH --job-name=mdf_ga_128core
#SBATCH --output=logs/mdf_ga_128core_%j.out
#SBATCH --error=logs/mdf_ga_128core_%j.err
#SBATCH --account=phys4840
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --time=48:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"

cd /project/joycelab-niall/MDF_GCE_GA || exit 1
source ~/python_projects/venv/bin/activate
mkdir -p logs

python MDF_GA.py

echo "Finished at: $(date)"
BATCH --job-name=mdf_ga_128core
#SBATCH --output=logs/mdf_ga_128core_%j.out
#SBATCH --error=logs/mdf_ga_128core_%j.err
#SBATCH --account=phys4840
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=24:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"

cd /project/joycelab-niall/MDF_GCE_GA || exit 1
source ~/python_projects/venv/bin/activate
mkdir -p logs

python MDF_GA.py

echo "Finished at: $(date)"

