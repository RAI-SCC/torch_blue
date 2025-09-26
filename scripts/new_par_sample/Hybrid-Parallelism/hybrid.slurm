#!/bin/bash

#SBATCH --job-name=Hybrid_Cifar10_CNN
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --mem=16G 
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --account=hai_1044
#SBATCH --output=/p/project1/hai_1044/oezdemir/sample_parallel/torch_bayesian/results/slurm-%j.out # Change to your output directory
#SBATCH --error=/p/project1/hai_1044/oezdemir/sample_parallel/torch_bayesian/results/error-%j.log          # Error log
#SBATCH --export=ALL



# Get the first node name as master address.
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12355
echo "MASTER_ADDR="$MASTER_ADDR
echo "MCD RESNET10 CIFAR10 4 GPUs 32 Samples"
echo $CUDA_VISIBLE_DEVICES


export PYDIR=/p/project1/hai_1044/oezdemir/sample_parallel/torch_bayesian # Set path to your python scripts.

module purge
module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3
python --version

# Activate environment
#source $PYDIR/newvenv/bin/activate


srun --ntasks=8 bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  echo "Rank $SLURM_PROCID using GPU $CUDA_VISIBLE_DEVICES on node $SLURM_NODEID"
  /p/project1/hai_1044/oezdemir/sample_parallel/torch_bayesian/venv/bin/python \
    -u /p/project1/hai_1044/oezdemir/sample_parallel/torch_bayesian/scripts/new_par_sample/Hybrid-Parallelism/hybrid_CNN_CIFAR10.py
'
