#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=1:0:0    
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:p100:1

cd $project/fair_imgs
module purge
module load python/3.10.11 scipy-stack
source ~/pyfair/bin/activate

python train_basemodel.py