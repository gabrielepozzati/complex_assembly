#!/bin/bash -x
#SBATCH -A berzelius-2021-64
#SBATCH --output=/proj/berzelius-2021-29/users/x_gabpo/job_out/AF%j.out
#SBATCH --error=/proj/berzelius-2021-29/users/x_gabpo/job_err/AF%j.err
#SBATCH --array=1-217
#SBATCH --gpus=1
#SBATCH -t 12:00:00

##### AF2 CONFIGURATION #####
COMMON="/proj/berzelius-2021-29/users/x_gabpo"
AFHOME="$COMMON/af2-v2.2.0"           		# Path of AF2-multimer-mod directory.
SINGULARITY="$AFHOME/data/af2.sif"		# Path of singularity image.
PARAM="$AFHOME/data/"                           # path of param folder containing AF2 Neural Net parameters.
MAX_RECYCLES=3
MODEL_SET="multimer"

### RUNTIME SPECIFICATIONS ###
LIST=$1
OFFSET=$2
FOLDER=$3

export NVIDIA_VISIBLE_DEVICES="all"
export TF_FORCE_UNIFIED_MEMORY="1"
export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"

POS=$(($SLURM_ARRAY_TASK_ID + $OFFSET))
PDB=`tail -n+$POS $LIST | head -n 1`

FILELIST=`ls ${FOLDER}${PDB} | grep "fasta" | grep "_"`

for FILE in $FILELIST; do
    cp $FOLDER/${PDB}/$FILE $FOLDER/${PDB}/${PDB}.fasta
    singularity exec --nv --bind $COMMON:$COMMON $SINGULARITY \
        python3 $AFHOME/run_alphafold.py \
            --fasta_paths=$FOLDER/${PDB}/${PDB}.fasta \
            --model_preset=$MODEL_SET \
            --output_dir=$FOLDER \
            --data_dir=$PARAM \
            --recycles=$MAX_RECYCLES \
            --db_preset=reduced_dbs \
            --num_multimer_predictions_per_model=1

    mkdir $FOLDER/${PDB}/${FILE:0:6}
    mv $FOLDER/${PDB}/${PDB}.fasta $FOLDER/${PDB}/${FILE:0:6}/
    mv $FOLDER/${PDB}/*.pdb $FOLDER/${PDB}/${FILE:0:6}/
    mv $FOLDER/${PDB}/*.pkl $FOLDER/${PDB}/${FILE:0:6}/
done;

