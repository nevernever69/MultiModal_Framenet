#!/bin/bash
#SBATCH --job-name=gemma_analysis
#SBATCH --partition=gpu                      # GPU partition (verify correct partition name)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                         # Request one GPU
#SBATCH --constraint=gpu2v100                # Request an H100 GPU (verify your system's flag)
#SBATCH --time=00:50:00                      # Adjust runtime as needed
#SBATCH --mem=64G
#SBATCH --output=/scratch/user/axb2032/MultiModal_Framenet/logs/%j.out
#SBATCH --error=/scratch/user/axb2032/Multimodal_framenet/logs/%j.err
#SBATCH --mail-user=axb2032@case.edu   # Replace with your email address
#SBATCH --mail-type=ALL

# --- Environment Setup ---
module load cuda/11.8
module load python/3.9

# Set the Hugging Face cache directory to your Gallina home (ensure this directory exists)
export TRANSFORMERS_CACHE=/home/galline/your_username/huggingface_cache
mkdir -p $TRANSFORMERS_CACHE

# Change to the scratch project directory
cd /scratch/your_username/gsoc2024-frame-blending

# Source configuration file (sets TMP_WORK_DIR, PROG_DIR, USERID, etc.)
source ./config.sh

# Activate your virtual environment (assumes it exists in the project directory)
source venv/bin/activate

# --- Run the Python Analysis Script ---
./new.sh
python gemma_analysis.py

# --- Transfer Results Back to Home Directory ---
mkdir -p /home/galline/your_username/gsoc2024-frame-blending/results
rsync -av /scratch/your_username/gsoc2024-frame-blending/results/ /home/galline/your_username/gsoc2024-frame-blending/results/

