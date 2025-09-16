#!/bin/bash
#SBATCH --job-name=GCE_task
#SBATCH --output=logs/GCE_task_%j.out
#SBATCH --error=logs/GCE_task_%j.err
#SBATCH --account=galacticbulge
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00








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


# --- resolve shared data roots (don’t guess: detect) ---
first_existing_dir() {
  for d in "$@"; do
    [ -d "$d" ] && { echo "$d"; return 0; }
  done
  return 1
}

PROJECT_DIR="/project/galacticbulge/MDF_GCE_GA"

# candidates for yield tables and iniab
YIELDS_DIR="$(first_existing_dir \
  "$PROJECT_DIR/yield_tables" \
  "$PROJECT_DIR/JINAPyCEE/yield_tables" \
  "$PROJECT_DIR/NuPyCEE/yield_tables")" || {
  echo "FATAL: could not find yield tables under {yield_tables, JINAPyCEE/yield_tables, NuPyCEE/yield_tables} in $PROJECT_DIR"
  exit 2
}

INIAB_DIR="$(first_existing_dir \
  "$PROJECT_DIR/iniabu" \
  "$PROJECT_DIR/yield_tables/iniabu" \
  "$PROJECT_DIR/JINAPyCEE/yield_tables/iniabu" \
  "$PROJECT_DIR/NuPyCEE/yield_tables/iniabu")" || {
  echo "FATAL: could not find iniabu directory under {iniabu, yield_tables/iniabu, JINAPyCEE/yield_tables/iniabu, NuPyCEE/yield_tables/iniabu}"
  exit 3
}

# verify the specific yield file referenced by your inlist actually exists
REQ_YIELD_FILE="agb_and_massive_stars_K10_LC18_Ravg.txt"
if [ ! -f "$YIELDS_DIR/$REQ_YIELD_FILE" ]; then
  echo "FATAL: $YIELDS_DIR/$REQ_YIELD_FILE not found."
  echo "Available yield files:"
  ls -1 "$YIELDS_DIR" | sed 's/^/  - /'
  exit 4
fi


YI="${YIELDS_DIR%/}/"; YI="${YI//\//\\/}"
II="${INIAB_DIR%/}/";  II="${II//\//\\/}"
OF="${PROJECT_DIR}/data/equal_weight_mdf.dat"; OF="${OF//\//\\/}"
OP="${PROJECT_DIR}/${RUN_DIR%/}/"; OP="${OP//\//\\/}"

sed -i "s|^sn1a_header:.*|sn1a_header: '${YI}'|"          bulge_pcard.txt
sed -i "s|^iniab_header:.*|iniab_header: '${II}'|"         bulge_pcard.txt
sed -i "s|^obs_file:.*|obs_file: '${OF}'|"                 bulge_pcard.txt
sed -i "s|^output_path:.*|output_path: '${OP}'|"           bulge_pcard.txt

# also set per-run knobs you’re sweeping
sed -i "s/^timesteps:.*/timesteps: ${TS}/"                 bulge_pcard.txt
sed -i "s/^mdf_vs_age_weight:.*/mdf_vs_age_weight: ${W}/"  bulge_pcard.txt
sed -i "s/^obs_age_data_target:.*/obs_age_data_target: '${TGT//\//\/}'/" bulge_pcard.txt

echo "[debug] sn1a_header=$(grep '^sn1a_header' bulge_pcard.txt)"
echo "[debug] iniab_header=$(grep '^iniab_header' bulge_pcard.txt)"
echo "[debug] obs_file=$(grep '^obs_file' bulge_pcard.txt)"
echo "[debug] output_path=$(grep '^output_path' bulge_pcard.txt)"


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
