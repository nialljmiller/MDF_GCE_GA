#!/bin/bash
set -euo pipefail

# grid
timesteps=(100)
weights=(1.0)
targets=(joyce)
attempt_no=(0 1)

mkdir -p logs

for at_no in "${attempt_no[@]}"; do
  for ts in "${timesteps[@]}"; do
    for w in "${weights[@]}"; do
      for tgt in "${targets[@]}"; do
        run_dir="bc_batch_medbow_${at_no}_${ts}_w_$(echo "$w*10" | bc | cut -d. -f1)_MDF"
        mkdir -p "$run_dir"
        sbatch --export=ALL,TS="$ts",W="$w",TGT="$tgt",RUN_DIR="$run_dir",RUN_NAME="t${ts}_w${w}_$(basename "$tgt")" launch_many.sh
      done
    done
  done
done
