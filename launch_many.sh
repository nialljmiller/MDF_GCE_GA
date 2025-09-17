#!/bin/bash
#SBATCH --job-name=GCE_task
#SBATCH --output=/project/galacticbulge/MDF_GCE_GA/logs/GCE_task_%j.out
#SBATCH --error=/project/galacticbulge/MDF_GCE_GA/logs/GCE_task_%j.err
#SBATCH --account=galacticbulge
#SBATCH --partition=mb                 # fastest start, preemptible
#SBATCH --qos=fast                     # 12h max; higher priority
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1                     # one Python proc
#SBATCH --cpus-per-task=96             # whole node
#SBATCH --exclusive                    # reserve the node
#SBATCH --requeue
#SBATCH --signal=B:USR1@120            # warning before preempt/time limit

set -euo pipefail

echo "Job: $SLURM_JOB_ID  Host: $(hostname)  Start: $(date)"

# ===== env passed from sbatch --export =====
: "${TS:?missing TS}"
: "${W:?missing W}"
: "${TGT:?missing TGT}"
: "${RUN_DIR:?missing RUN_DIR}"
RUN_NAME="${RUN_NAME:-t${TS}_w${W}_$(basename "$TGT" | tr -c 'A-Za-z0-9._-' '_')}"
# ==========================================

PROJECT_DIR="/project/galacticbulge/MDF_GCE_GA"
PYENV="${PYENV:-$HOME/python_projects/venv}"
source "$PYENV/bin/activate"

# whole-node threading
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

mkdir -p "$PROJECT_DIR/logs"
JOB_WORKDIR="$PROJECT_DIR/work/$RUN_NAME"
mkdir -p "$JOB_WORKDIR"
cp -f "$PROJECT_DIR/bulge_pcard.txt" "$JOB_WORKDIR/bulge_pcard.txt"
pushd "$JOB_WORKDIR" >/dev/null

# --- preemption trap → forward to python, requeue, exit 0 ---
on_preempt() {
  echo "[preempt] $(date) USR1 received; forwarding to python and requeueing..."
  pkill -USR1 -P "$$" || true
  sleep 30
  scontrol requeue "$SLURM_JOB_ID" || true
  exit 0
}
trap on_preempt USR1

# --- locate shared data robustly ---
first_existing_dir() { for d in "$@"; do [ -d "$d" ] && { echo "$d"; return 0; }; done; return 1; }

YIELDS_DIR="$(first_existing_dir \
  "$PROJECT_DIR/yield_tables" \
  "$PROJECT_DIR/JINAPyCEE/yield_tables" \
  "$PROJECT_DIR/NuPyCEE/yield_tables")" || { echo "FATAL: yield_tables not found"; exit 2; }

INIAB_DIR="$(first_existing_dir \
  "$PROJECT_DIR/iniabu" \
  "$PROJECT_DIR/yield_tables/iniabu" \
  "$PROJECT_DIR/JINAPyCEE/yield_tables/iniabu" \
  "$PROJECT_DIR/NuPyCEE/yield_tables/iniabu")" || { echo "FATAL: iniabu not found"; exit 3; }


OBS_FILE="$PROJECT_DIR/data/equal_weight_mdf.dat"
[ -f "$OBS_FILE" ] || { echo "FATAL: $OBS_FILE missing"; exit 4; }

# --- rewrite inlist with ABS paths + per-run knobs ---
yi="${YIELDS_DIR//\//\\/}/"
ii="${INIAB_DIR//\//\\/}/"
of="${OBS_FILE//\//\\/}"
out="${PROJECT_DIR//\//\\/}/${RUN_DIR//\//\\/}/"

sed -i "s|^sn1a_header:.*|sn1a_header: '${yi}'|"          bulge_pcard.txt
sed -i "s|^iniab_header:.*|iniab_header: '${ii}'|"         bulge_pcard.txt
sed -i "s|^obs_file:.*|obs_file: '${of}'|"                 bulge_pcard.txt
sed -i "s|^output_path:.*|output_path: '${out}'|"          bulge_pcard.txt
sed -i "s/^timesteps:.*/timesteps: ${TS}/"                 bulge_pcard.txt
sed -i "s/^mdf_vs_age_weight:.*/mdf_vs_age_weight: ${W}/"  bulge_pcard.txt
sed -i "s/^obs_age_data_target:.*/obs_age_data_target: '${TGT//\//\/}'/" bulge_pcard.txt

echo "[inlist] $(grep -E '^(sn1a_header|iniab_header|obs_file|output_path|timesteps|mdf_vs_age_weight|obs_age_data_target):' bulge_pcard.txt)"

# --- run one process using all cores ---
srun --cpu-bind=cores -n 1 python "$PROJECT_DIR/MDF_GA.py"

popd >/dev/null
echo "Finish: $(date)"
