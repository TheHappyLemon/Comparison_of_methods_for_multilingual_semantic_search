import h5py
from constants import *

# File used for merging data of multiple hdf5 files into a single one.

def copy_dataset(source_file_path, source_dataset_path, target_file_path, target_dataset_path):
    with h5py.File(source_file_path, 'r') as source_file:
        if source_dataset_path in source_file:
            with h5py.File(target_file_path, 'a') as target_file:
                data = source_file[source_dataset_path]
                target_file.create_dataset(target_dataset_path, data=data)
                print(f"Data copied from {source_dataset_path} to {target_dataset_path} in {target_file_path}")
        else:
            print(f"Dataset {source_dataset_path} not found in {source_file_path}")

import h5py

def copy_data_range(source_file_path, source_dataset_path, target_file_path, target_dataset_path, start_index, end_index):
    with h5py.File(source_file_path, 'r') as source_file:
        with h5py.File(target_file_path, 'a') as target_file:
            if not target_dataset_path in target_file:
                target_file.create_dataset(target_dataset_path, shape=(77736, 1024))
            for i in range(start_index, end_index):     
                target_file[target_dataset_path][i] = source_file[source_dataset_path][i]
                if i % 500 == 0:
                    print(i)

source_file = path_res + 'embeddings_LASER_en_source_from69.hdf5'
target_file = path_res + 'embeddings.hdf5'
dataset = '/LASER/source/en_cirrussearch'
start_index = 69000  # Starting index (inclusive)
end_index = 77736    # Ending index (exclusive)

copy_data_range(source_file, dataset, target_file, dataset, start_index, end_index)


#source_file = path_res + 'embeddings_LASER_backup_8.hdf5'
#target_file =  path_res + 'embeddings.hdf5'
#datasets = ['/LASER/open/en_cirrussearch', '/LASER/open/lv_cirrussearch', '/LASER/title/en_cirrussearch', '/LASER/title/lv_cirrussearch']
#for dataset in datasets:
    #copy_dataset(source_file, dataset, target_file, dataset)
    #print(dataset, '->', 'done')
