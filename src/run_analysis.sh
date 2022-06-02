#!/bin/bash -x
#SBATCH -A berzelius-2022-106
#SBATCH --output=/proj/berzelius-2021-29/users/x_gabpo/job_out/AF%j.out
#SBATCH --error=/proj/berzelius-2021-29/users/x_gabpo/job_err/AF%j.err
#SBATCH --array=1-10
#SBATCH -n 1
#SBATCH -t 01:00:00

### RUNTIME SPECIFICATIONS ###
LIST=$1
OFFSET=$2

POS=$(($SLURM_ARRAY_TASK_ID + $OFFSET))
PDB=`tail -n+$POS $LIST | head -n 1`

mkdir /proj/berzelius-2021-29/users/x_gabpo/tmp/tmp_${PDB}/
singularity exec --nv --bind /proj/berzelius-2021-29/users/x_gabpo:/proj/berzelius-2021-29/users/x_gabpo /proj/berzelius-2021-29/users/x_gabpo/af2-v2.2.0/data/af2.sif python3 /proj/berzelius-2021-29/users/x_gabpo/complex_assembly/src/analyze_dockings.py $PDB
