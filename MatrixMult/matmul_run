#!/bin/bash

# Do not forget to select a proper partition if
# the default one is no fit for the job!

#SBATCH --output=./logs/out.%j
#SBATCH --error=./logs/err.%j
#SBATCH --nodes=1          # number of nodes
# SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --exclusive
#SBATCH --time=00:30:00    # walltime
#SBATCH --gres=gpu:1
#SBATCH --reservation=pram 

# Stop operation on first error.
set -e

# Load environment modules for your application here.
source /etc/profile.d/modules.sh
module purge
module load cuda/11.6

# Actual work starting here.

echo "cuBLAS Single Precision Matrix Multiplication"
echo "========================================="
srun ./mm_cuBlas_float

echo ""

echo "CUDA Single Precision Matrix Multiplication"
echo "========================================="
srun ./mm_cuda_float

echo ""

echo "cuBLAS Double Precision Matrix Multiplication"
echo "========================================="
#srun ./mm_cuBlas_double

echo ""

echo "CUDA Double Precision Matrix Multiplication"
echo "========================================="
#srun ./mm_cuda_double
