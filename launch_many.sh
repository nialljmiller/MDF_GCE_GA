#!/bin/bash
#SBATCH --job-name=GCE_task
#SBATCH --output=logs/GCE_task_%j.out
#SBATCH --error=logs/GCE_task_%j.err
#SBATCH --account=galacticbulge
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

# --- REQUIRED ENV VARS (exported by the dispatcher) ---
TS="${TS:?missing TS (timesteps)}"
W="${W:?missing W (mdf_vs_age_weight)}"
TGT="${TGT:?missing TGT (obs_age_data_target)}"
RUN_DIR="${RUN_DIR:?missing RUN_DIR (output_path target)}"

PROJECT_DIR="/project/galacticbulge/MDF_GCE_GA"
cd "$PROJECT_DIR"

# Python env
source ~/python_projects/venv/bin/activate

# Each job gets its own working dir and its own bulge_pcard.txt
RUN_NAME="${RUN_NAME:-t${TS}_w${W}_$(basename "$TGT" | tr -c 'A-Za-z0-9._-' '_')}"
JOB_WORKDIR="$PROJECT_DIR/work/$RUN_NAME"

mkdir -p "$JOB_WORKDIR" "logs"  # make sure logs/ exists for SBATCH outputs
cp "$PROJECT_DIR/bulge_pcard.txt" "$JOB_WORKDIR/bulge_pcard.txt"

# Modify the LOCAL copy of the param file (avoid races on shared file)
pushd "$JOB_WORKDIR" >/dev/null

sed -i "s/^timesteps:.*/timesteps: ${TS}/" bulge_pcard.txt
sed -i "s/^mdf_vs_age_weight:.*/mdf_vs_age_weight: ${W}/" bulge_pcard.txt
sed -i "s/^obs_age_data_target:.*/obs_age_data_target: '${TGT//\//\/}'/" bulge_pcard.txt
# Ensure trailing slash on output_path in the param file
OUTP="${RUN_DIR%/}/"
sed -i "s|^output_path:.*|output_path: '${OUTP//'/'\/}'|" bulge_pcard.txt

echo "Params -> timesteps=${TS} weight=${W} target=${TGT} out=${OUTP}"

# Run the code (one task per job)
srun -n 1 python "$PROJECT_DIR/MDF_GA.py"

popd >/dev/null

echo "Finish: $(date)"
