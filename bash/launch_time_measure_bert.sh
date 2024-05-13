#!/bin/bash

#PBS -N measure_time_bert
#PBS -q batch
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=16
#PBS -j oe
#PBS -W x=HOSTLIST:wn64

INP1="$HOME/repos/Comparison_of_methods_for_multilingual_semantic_search/Dataset"
DIR="/scratch/artjom01/"
DIRR="/scratch/artjom01"
if [ ! -d "$DIR" ]; then
    mkdir "$DIR"
    echo "Creating directory: $DIR"
fi

# Copy wikipedia pages to local node drive
#cp -r $INP1 $DIR 
#cp -r $INP2 $DIR

# load anaconda with my environment
module load anaconda/conda-23.1.0
source activate my_conda_env

# laucnh script to generate embeddings
python3 $HOME/repos/Comparison_of_methods_for_multilingual_semantic_search/Code/measure_time.py bert title en 10000
python3 $HOME/repos/Comparison_of_methods_for_multilingual_semantic_search/Code/measure_time.py bert title lv 10000
python3 $HOME/repos/Comparison_of_methods_for_multilingual_semantic_search/Code/measure_time.py bert open en 10000
python3 $HOME/repos/Comparison_of_methods_for_multilingual_semantic_search/Code/measure_time.py bert open lv 10000
python3 $HOME/repos/Comparison_of_methods_for_multilingual_semantic_search/Code/measure_time.py bert source en 10000
python3 $HOME/repos/Comparison_of_methods_for_multilingual_semantic_search/Code/measure_time.py bert source lv 10000

