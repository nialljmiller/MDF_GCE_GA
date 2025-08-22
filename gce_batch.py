import os
import shutil
import subprocess
import sys
import fileinput
import numpy as np  # Import numpy for generating ranges if needed

# Base parameter file (assumed to exist and contain all keys, even if dummy values)
BASE_PARAM_FILE = 'bulge_pcard.txt'

# Main script to run (assumed to parse 'bulge_pcard.txt')
MAIN_SCRIPT = 'MDF_GA.py'

# Function to modify parameter file in place (replaces lines starting with 'key:')
def modify_param_file(input_file, modifications):
    with fileinput.FileInput(input_file, inplace=True) as file:
        for line in file:
            replaced = False
            for key, value in modifications.items():
                if line.strip().startswith(key + ':'):
                    print(f"{key}: {value}")
                    replaced = True
                    break
            if not replaced:
                print(line, end='')

# Function to generate unique folder name based on params
def generate_folder_name(timestep, weight, target):
    return f"bc_batch_{timestep}_w_{int(weight*10)}_{target.lower()}"

# Main wrapper logic (hardcoded values instead of args)
def main():
    # Hardcode the grid values here
    # Option 1: Manually enter lists
    timesteps = [10, 50]  # Manually defined timesteps
    weights = [0.0, 0.5, 1.0]   # Manually defined weights
    targets = ['joyce', 'bensby']  # Manually defined targets

    # Option 2: Use numpy to generate (uncomment and adjust as needed)
    # timesteps = np.linspace(50, 200, num=3).astype(int).tolist()  # Generates [50, 125, 200]
    # weights = np.linspace(0.3, 0.7, num=3).tolist()  # Generates [0.3, 0.5, 0.7]
    # targets = ['joyce', 'bensby']  # Targets remain manual (strings)

    # Grid loop (nested for Cartesian product)
    for timestep in timesteps:
        for weight in weights:
            for target in targets:
                # Generate unique run dir
                run_dir = generate_folder_name(timestep, weight, target)
                
                # Create run directory if needed
                os.makedirs(run_dir, exist_ok=True)
                
                # Define modifications (add more keys as needed)
                modifications = {
                    'timesteps': timestep,
                    'mdf_vs_age_weight': weight,
                    'obs_age_data_target': f"'{target}'",  # Assumes string format in param file
                    'output_path': f"'{run_dir}/'"  # Set output_path to this dir
                }
                
                # Create modified param file in run_dir
                modify_param_file(BASE_PARAM_FILE, modifications)
                
                # Change to run_dir and execute the main script
                cwd = os.getcwd()
                try:
                    #os.chdir(run_dir)
                    print(f"Starting run in {run_dir} with params: ts={timestep}, w={weight}, target={target}")
                    subprocess.run([sys.executable, os.path.join(cwd, MAIN_SCRIPT)], check=True)
                    print(f"Completed run in {run_dir}")
                except subprocess.CalledProcessError as e:
                    print(f"Error in run {run_dir}: {e}")
                finally:
                    os.chdir(cwd)

if __name__ == '__main__':
    main()