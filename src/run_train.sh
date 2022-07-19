#!/bin/bash -x
#SBATCH -A berzelius-2022-106
#SBATCH --output=/proj/berzelius-2021-29/users/x_gabpo/job_out/trainAE_%j.out
#SBATCH --error=/proj/berzelius-2021-29/users/x_gabpo/job_err/trainAE_%j.err
#SBATCH --array=1-1
#SBATCH -n 1
#SBATCH -t 01:00:00

### RUNTIME SPECIFICATIONS ###
cd /proj/berzelius-2021-29/users/x_gabpo/complex_assembly/
singularity exec --nv --bind /proj/berzelius-2021-29/users/x_gabpo:/proj/berzelius-2021-29/users/x_gabpo ../af2-v2.2.0/data/af2 python3 src/train_ae.py
