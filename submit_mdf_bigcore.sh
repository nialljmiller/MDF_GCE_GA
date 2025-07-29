#!/bin/bash
#SBATCH --job-name=mdf_ga_96core
#SBATCH --output=logs/mdf_ga_128core_%j.out
#SBATCH --error=logs/mdf_ga_128core_%j.err
#SBATCH --account=galacticbulge
#SBATCH --nodes=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --time=96:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"

cd /project/galacticbulge/MDF_GCE_GA || exit 1
source ~/python_projects/venv/bin/activate
mkdir -p logs
python -u MDF_GA.py

echo "Finished at: $(date)"

