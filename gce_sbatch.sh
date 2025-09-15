#!/bin/bash
set -euo pipefail

# Optional: backup of the baseline param file (not modified by this script anymore)
cp -f bulge_pcard.txt bulge_pcard_backup.txt

# Parameter grids
timesteps=(100)
weights=(1.0)
targets=(joyce)
attempt_no=(0 1 2 3 4)

mkdir -p logs

# Loop over combinations
for at_no in "${attempt_no[@]}"; do
  for ts in "${timesteps[@]}"; do
    for w in "${weights[@]}"; do
      for tgt in "${targets[@]}"; do
        run_dir="bc_batch_local_${at_no}_${ts}_w_$(echo "$w * 10" | bc | cut -d. -f1)_${tgt,,}"
        mkdir -p "$run_dir"
        sbatch --export=ALL,TS="$ts",W="$w",TGT="$tgt",RUN_DIR="$run_dir",RUN_NAME="t${ts}_w${w}_$(basename "$tgt")" launch_many.sh
      done
    done
  done
done
