#!/bin/bash
#SBATCH -n 1
#SBATCH -p hugheslab
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH -o /cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out # Write stdout to file named log_JOBIDNUM.out in log dir
#SBATCH -e /cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err # Write stderr to file named log_JOBIDNUM.err in log dir
#SBATCH --array=0-2

source ~/.bashrc
conda activate bdl_2022f_env

# Define an array of commands
experiments=(
    "python ../src/finetune_3D.py --attention --experiments_path='/cluster/tufts/hugheslab/eharve06/brain-scan-classifiers/experiments/Pre-Trained_ViT_OASIS-3_MRI_new' --labels_path='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_MRI/random_state=1001' --max_slices=169 --n=1300 --random_state=1001"
    "python ../src/finetune_3D.py --attention --experiments_path='/cluster/tufts/hugheslab/eharve06/brain-scan-classifiers/experiments/Pre-Trained_ViT_OASIS-3_MRI_new' --labels_path='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_MRI/random_state=2001' --max_slices=169 --n=1300 --random_state=2001"
    "python ../src/finetune_3D.py --attention --experiments_path='/cluster/tufts/hugheslab/eharve06/brain-scan-classifiers/experiments/Pre-Trained_ViT_OASIS-3_MRI_new' --labels_path='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_MRI/random_state=3001' --max_slices=169 --n=1300 --random_state=3001"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate