#!/bin/bash
#SBATCH --job-name=oct
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200|l40s"
#SBATCH --comment="gpu_mps=yes"
#SBATCH --mem=60GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=gbc9195@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --account=torch_pr_226_chemistry

module purge;
module load anaconda3/2025.06;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export PATH_TO_ENV=/scratch/gbc9195/conda_envs/openmm-cuda11;
source activate $PATH_TO_ENV;
export PATH=$PATH_TO_ENV/bin:$PATH;

(
  cd replicate-1
  python run_lambda_OPES_AFED.py
) &

(
  cd replicate-2
  python run_lambda_OPES_AFED.py
) &

(
  cd replicate-3
  python run_lambda_OPES_AFED.py
) &

wait
