#!/bin/bash
#SBATCH -n 6
#SBATCH -N 1
#SBATCH -D /fhome/vlia01/Medical-Imaging/slurm_output/
#SBATCH -t 4-00:05
#SBATCH -p tfg
#SBATCH --mem 12288
#SBATCH --gres gpu:1

#export WANDB_API_KEY=29cf4fb380fd7bca6b40de8bab6b441105ac51f4
export WANDB_API_KEY=786350642110ef7f9f59ffa010eb177dc549bd7c

sleep 3

python3 ../train_conv_ae.py --config_number 1