#!/bin/bash
#SBATCH --job-name=hostname 
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/slurm/R-%x.%j.out
python mymain.py model.epochs=1
