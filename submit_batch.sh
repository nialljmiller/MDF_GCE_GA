#!/bin/bash
#SBATCH --job-name=batch_GCE
#SBATCH --output=logs/batch_GCE_%j.out
#SBATCH --error=logs/batch_GCE_%j.err
#SBATCH --account=galacticbulge
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=96:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"

cd /project/galacticbulge/MDF_GCE_GA || exit 1
source ~/python_projects/venv/bin/activate
mkdir -p logs
./gce_batch.sh

echo "Finished at: $(date)"


