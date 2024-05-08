#!/bin/bash

# Commands to sync setup and Dataset/en_cirrussearch_source folders to multiple servers

# Syncing setup folder
rsync -r -e ssh setup/ artjom01@wn61:/scratch/artjom01/setup/
rsync -r -e ssh setup/ artjom01@wn62:/scratch/artjom01/setup/
rsync -r -e ssh setup/ artjom01@wn67:/scratch/artjom01/setup/
rsync -r -e ssh setup/ artjom01@wn68:/scratch/artjom01/setup/
rsync -r -e ssh setup/ artjom01@wn69:/scratch/artjom01/setup/
rsync -r -e ssh setup/ artjom01@wn70:/scratch/artjom01/setup/

# Syncing Dataset/en_cirrussearch_source folder
rsync -r -e ssh Dataset/en_cirrussearch_source/ artjom01@wn62:/scratch/artjom01/Dataset/en_cirrussearch_source/
rsync -r -e ssh Dataset/en_cirrussearch_source/ artjom01@wn67:/scratch/artjom01/Dataset/en_cirrussearch_source/
rsync -r -e ssh Dataset/en_cirrussearch_source/ artjom01@wn68:/scratch/artjom01/Dataset/en_cirrussearch_source/
rsync -r -e ssh Dataset/en_cirrussearch_source/ artjom01@wn69:/scratch/artjom01/Dataset/en_cirrussearch_source/
rsync -r -e ssh Dataset/en_cirrussearch_source/ artjom01@wn70:/scratch/artjom01/Dataset/en_cirrussearch_source/

echo "Sync complete for both setup and Dataset/en_cirrussearch_source folders!"

