#!/bin/bash
#SBATCH -N 1
#SBATCH -p small-gpu
#SBATCH -t 2-00:00:00 
#SBATCH --mail-type=begin 
#SBATCH --mail-type=end  


srun python main.py --Data CAR --Esize 12 --Msize 12 --epochs 12
