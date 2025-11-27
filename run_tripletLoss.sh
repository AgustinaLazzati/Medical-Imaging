#!/bin/bash
#SBATCH -n 6
#SBATCH -N 1
#SBATCH -D /fhome/vlia01/Medical-Imaging/slurm_output/
#SBATCH -t 0:30:00
#SBATCH -p tfg
#SBATCH --mem 32288
#SBATCH --gres gpu:1


### AGU'S KEY:
export WANDB_API_KEY=29cf4fb380fd7bca6b40de8bab6b441105ac51f4

### TOMI'S KEY:
##export WANDB_API_KEY=786350642110ef7f9f59ffa010eb177dc549bd7c

sleep 3

python3 ../tripletLoss.py