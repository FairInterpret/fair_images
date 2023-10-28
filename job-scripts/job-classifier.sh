#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=15:0:0    
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

cd $project/hpc-fairimgs
module purge
module load python/3.10.11 scipy-stack
source ~/fairimg/bin/activate

python main/train_classification.py