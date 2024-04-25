INITIAL SETUP:
1) install conda (miniconda) here: https://docs.conda.io/projects/conda/en/stable/
2) add to enviroment: https://stackoverflow.com/questions/44515769/conda-is-not-recognized-as-internal-or-external-command
3) install faiss - conda install -c pytorch faiss-cpu=1.8.0 https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
4) conda install python=3.11 - to downgrade because navigator does not isntall with python 3.12
5) conda install anaconda-navigator 
6) can open anaconda navigator

additional:
conda install matplotlib
conda install transformers
conda config --append channels conda-forge - to install torch
conda install -c conda-forge pytorch - installs pytorch! 
----------------------------------------------------------------------------------------------------------------------------
                                            laser_encoders

laser_encoders is not directly available through Anaconda's main repository, so => install via pip:
conda install pip
pip install laser-encoders

laser_encoders depends on fairseq, which is not compatible with python 3.11.8 !!! I had to switch to python 3.9 to make it work. 
https://github.com/facebookresearch/fairseq/issues/5191
https://github.com/facebookresearch/LASER/issues/280
Also had to fix download script - https://github.com/facebookresearch/LASER/pull/282
