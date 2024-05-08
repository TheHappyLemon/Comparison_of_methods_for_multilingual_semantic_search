#!/bin/bash

#PBS -N gen_embd_LASER
#PBS -q batch
#PBS -l walltime=90:00:00
#PBS -l nodes=1:ppn=12:gpus=1:shared
#PBS -j oe
#PBS -W x=HOSTLIST:wn63

INP1="$HOME/Comparison_of_methods_for_multilingual_semantic_search/Dataset"
INP2="$HOME/Comparison_of_methods_for_multilingual_semantic_search/setup"
DIR="/scratch/artjom01/"
DIRR="/scratch/artjom01"
if [ ! -d "$DIR" ]; then
    mkdir "$DIR"
    echo "Creating directory: $DIR"
fi

# Copy wikipedia pages to local node drive
cp -r $INP1 $DIR 
cp -r $INP2 $DIR

# load anaconda with my environment
module load anaconda/conda-23.1.0
source activate my_conda_env

# laucnh script to generate embeddings
python3 $HOME/Comparison_of_methods_for_multilingual_semantic_search/Code/generate_embeddings_LASER.py
cp "$DIRR/Dataset/embeddings.hdf5" "$HOME/embeddings_result_laser.hdf5"
cp "$DIRR/generate_embeddings_LASER.log" "$HOME/generate_embeddings_LASER_result.log"
