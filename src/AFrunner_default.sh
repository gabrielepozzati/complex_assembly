#!/bin/bash -x
#SBATCH -A SNIC2021-5-297
#SBATCH --output=/proj/snic2019-35-62/users/x_gabpo/job_out/hh%j.out
#SBATCH --error=/proj/snic2019-35-62/users/x_gabpo/job_err/hh%j.err
#SBATCH --array=1-1
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gpus-per-task=1
#SBATCH -t 12:00:00

##### AF2 CONFIGURATION #####
COMMON="/proj/snic2019-35-62/"
AFHOME="$COMMON/af2-v2.2.0"           		        # Path of AF2-multimer-mod directory.
DATAHOME="$COMMON/users/x_gabpo"			# Path of data folder
SINGULARITY="$AFHOME/AF_data_v220/alphafold_v220.sif"   # Path of singularity image.
PARAM="$AFHOME/AF_data_v220/"                           # path of param folder containing AF2 Neural Net parameters.
MAX_RECYCLES=3
MODEL_SET="multimer"

### RUNTIME SPECIFICATIONS ###
LIST=$1
OFFSET=$2
SUBFOLDER=$3 # Subpath of $DATAHOME where fasta and a3m are stored

DATA="${DATAHOME}/${SUBFOLDER}/"

POS=$(($SLURM_ARRAY_TASK_ID + $OFFSET))
PDB=`tail -n+$POS $LIST | head -n 1`
echo ${DATA}${PDB}

singularity exec --nv --bind $COMMON:$COMMON \
    --env="NVIDIA_VISIBLE_DEVICES=all" \
    --env="TF_FORCE_UNIFIED_MEMORY=1" \
    --env="XLA_PYTHON_CLIENT_MEM_FRACTION=4.0" \
    --env="OPENMM_CPU_THREADS=32" \
    --env="MAX_CPUS=32" \
    $SINGULARITY \
    python3 $AFHOME/alphafold/run_alphafold.py \
        --fasta_paths=$DATA/${PDB}.fasta \
        --model_preset=$MODEL_SET \
        --output_dir=$DATA \
        --data_dir=$PARAM \
	--recycles=$MAX_RECYCLES \
        --db_preset=reduced_dbs \
        --num_multimer_predictions_per_model=1

