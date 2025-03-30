#!/bin/bash
# config.sh
export TMP_WORK_DIR="/scratch/users/axb2032"
export PROG_DIR="MultiModal_Framenet"
export USERID="axb2032"

#module load Python/3.11.3
#module load CUDA/12.3.0
#module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
#module load PyYAML/6.0-GCCcore-12.3.0
# Purge any existing modules

# Load a recent GCC version (required for CUDA 12+)
module load GCC/11.2.0

# Load the CUDA toolkit module that supports H100 (e.g. CUDA 12.1 or newer)
module load CUDA/12.1

# Load cuDNN (matching your CUDA version)
module load   cuDNN/8.9.2.26-CUDA-12.1.1

# Load NCCL for multi-GPU communication (if required)

# Load CMake (make sure it is a version that supports your CUDA toolkit)
module load    CMake/3.24.3-GCCcore-12.3.0

# Load Python (if your project requires it)
module load Python/3.11.3
