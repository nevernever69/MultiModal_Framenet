#!/bin/bash
#SBATCH --job-name=gemma_analysis
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                         # Request one GPU
#SBATCH --partition=gpu                    # Specify correct partition
#SBATCH --constraint=gpu2h100                # Request an H100 GPU (verify your system's flag)
#SBATCH --time=05:00:00                      # Adjust runtime as needed
#SBATCH --mem=64G
#SBATCH --output=/scratch/users/axb2032/MultiModal_Framenet/logs/%j.out
#SBATCH --error=/scratch/users/axb2032/MultiModal_Framenet/logs/%j.err
#SBATCH --mail-user=axb2032@case.edu   # Replace with your email address
#SBATCH --mail-type=ALL

# --- Environment Setup ---

source ./config.sh
# Change to the scratch project directory
cd $TMP_WORK_DIR/$PROG_DIR

# Source configuration file (sets TMP_WORK_DIR, PROG_DIR, USERID, etc.)
# Activate your virtual environment (assumes it exists in the project directory)
source venv/bin/activate

# --- Run the Python Analysis Script ---
#./Download_dataset_extract.sh
python Gemma_parallel_GPU.py

# --- Transfer Results Back to Home Directory ---
#rsync -av
