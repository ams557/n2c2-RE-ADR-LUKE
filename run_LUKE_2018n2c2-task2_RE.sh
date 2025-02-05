#!/bin/bash

## Begin SLURM Batch Commands
#SBATCH --cpus-per-task=31
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100
#SBATCH --mem=100G
#SBATCH --time=0-08:00
#SBATCH --mail-user shapiroa2@vcu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=LUKE.log

module load miniconda3
conda activate py311
srun python -m src.trainer

## ** End Of SLURM Batch Commands **
##
## ===================================
## Important Hickory GPU Request Note
## ===================================
## Most importantly, the option `--gres=gpu:<type>:<count>` must be used
## to request GPUs (`-G` or `--gpus` will not work). Values for `<type>`
## are `40g` and `80g`, referring to the 40 GB and 80 GB GPUs. The current
## limits (`<count>`) for the 40 GB GPUs are 1 in the `long` QOS and 2 in
## `short`. The current limit for the 80 GB GPUs is 1 in `short` (they are
## unavailable in `long`).
##
##
## More Info: https://wiki.vcu.edu/x/P6POBQ
##
## END