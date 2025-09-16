#!/bin/bash
#SBATCH --job-name=GCE_task
#SBATCH --output=logs/GCE_task_%j.out
#SBATCH --error=logs/GCE_task_%j.err
#SBATCH --account=galacticbulge
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2:00:00

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

# --- REQUIRED (from dispatcher) ---
TS="${TS:?missing TS}"
W="${W:?missing W}"
TGT="${TGT:?missing TGT}"
RUN_DIR="${RUN_DIR:?missing RUN_DIR}"
RUN_NAME="${RUN_NAME:-t${TS}_w${W}_$(basename "$TGT" | tr -c 'A-Za-z0-9._-' '_')}"

# Project root (adjust if needed)
PROJECT_DIR="/project/galacticbulge/MDF_GCE_GA"

# Python env
source ~/python_projects/venv/bin/activate

# One process uses all CPUs we requested
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Private working dir per job
JOB_WORKDIR="$PROJECT_DIR/work/$RUN_NAME"
mkdir -p "$JOB_WORKDIR" "$PROJECT_DIR/logs"
cp "$PROJECT_DIR/bulge_pcard.txt" "$JOB_WORKDIR/bulge_pcard.txt"

pushd "$JOB_WORKDIR" >/dev/null

# (A) Make data paths absolute inside the copied param file
# Your param keys (relative in the repo): obs_file, iniab_header, sn1a_header.
# We rewrite them to absolute paths under $PROJECT_DIR so running from JOB_WORKDIR works.
sed -i "s|^obs_file:.*|obs_file: '$PROJECT_DIR/data/equal_weight_mdf.dat'|" bulge_pcard.txt
sed -i "s|^iniab_header:.*|iniab_header: '$PROJECT_DIR/yield_tables/iniabu/'|" bulge_pcard.txt
sed -i "s|^sn1a_header:.*|sn1a_header: '$PROJECT_DIR/yield_tables/'|" bulge_pcard.txt

# Also set the per-run knobs
sed -i "s/^timesteps:.*/timesteps: ${TS}/" bulge_pcard.txt
sed -i "s/^mdf_vs_age_weight:.*/mdf_vs_age_weight: ${W}/" bulge_pcard.txt
sed -i "s/^obs_age_data_target:.*/obs_age_data_target: '${TGT//\//\/}'/" bulge_pcard.txt

# Ensure a unique output_path (with trailing slash)
OUTP="${PROJECT_DIR}/${RUN_DIR%/}/"
sed -i "s|^output_path:.*|output_path: '${OUTP//'/'\/}'|" bulge_pcard.txt

echo "Params -> timesteps=${TS} weight=${W} target=${TGT} out=${OUTP}"

# Optional: quick sanity
python - <<'PY'
from pathlib import Path
p = Path("bulge_pcard.txt").read_text()
for k in ("obs_file","iniab_header","sn1a_header","output_path"):
    for line in p.splitlines():
        if line.strip().startswith(f"{k}:"):
            print(line)
PY

# Run one Python process that will use all CPUs on this node
# (Your repo’s entrypoint is MDF_GA.py; keep that.)
srun -n 1 python "$PROJECT_DIR/MDF_GA.py"

popd >/dev/null
echo "Finish: $(date)"
