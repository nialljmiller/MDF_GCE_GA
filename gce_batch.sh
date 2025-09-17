#!/bin/bash

# Backup original param file
cp bulge_pcard.txt bulge_pcard_backup.txt

# Function to modify param file
modify_param() {
  sed -i "s/^timesteps:.*/timesteps: $1/" bulge_pcard.txt
  sed -i "s/^mdf_vs_age_weight:.*/mdf_vs_age_weight: $2/" bulge_pcard.txt
  sed -i "s/^obs_age_data_target:.*/obs_age_data_target: '$3'/" bulge_pcard.txt
  sed -i "s/^output_path:.*/output_path: '$4\/'/" bulge_pcard.txt
}

# Parameter grids
timesteps=(10)
weights=(0.0)
targets=(joyce)
attempt_no=(0 1 2 3 4)

# Loop over combinations

for at_no in "${attempt_no[@]}"; do
  for ts in "${timesteps[@]}"; do
    for w in "${weights[@]}"; do
      for tgt in "${targets[@]}"; do
        run_dir="bc_batch_local_${at_no}_${ts}_w_$(echo "$w * 10" | bc | cut -d. -f1)_MDF"
        mkdir -p "$run_dir"

        # Modify param file
        modify_param "$ts" "$w" "$tgt" "$run_dir"

        # Echo and run command
        cmd="python MDF_GA.py"
        echo "Running: $cmd (for $run_dir)"
        $cmd

        # Restore original
        cp bulge_pcard_backup.txt bulge_pcard.txt
        echo "Completed $run_dir"
      done
    done
  done
done