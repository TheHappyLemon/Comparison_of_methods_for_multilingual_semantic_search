import h5py
from constants import path_res, path_search, path_log, chunk_size, path_search_FLAT
from utils import get_np_array_zero_rows, get_dict_from_json
import faiss
import numpy as np
import csv
import os
from io import TextIOWrapper
from datetime import datetime

csv_header = ['query_file', 'found', 'search_result', 'search_distances']
ignore_zeroes = True

def build_index_from_hdf(file_path : str, dataset : str) -> faiss.IndexHNSWFlat:
    with h5py.File(file_path, 'r') as file:

        if not dataset in file:
            raise ValueError(f"Dataset {dataset} not in file")
        # np.any - non zero rows, ~ - invert, np.where - return array of indexes with True values
        zero_rows_indexes = get_np_array_zero_rows(file[dataset])
        if len(zero_rows_indexes) != 0:
            if ignore_zeroes:
                #.item() -> convert numpy type to python native
                limit = zero_rows_indexes[0].item()
            else:
                raise ValueError(f"Embedding in dataset '{dataset}' on index '{zero_rows_indexes[0]}' is not calculated!!!")
        else:
            limit = file[dataset].shape[0]
        
        index = faiss.IndexFlatL2(file[dataset].shape[1])
        #index = faiss.IndexHNSWFlat(file[dataset].shape[1], 64)
        index.add(file[dataset][:limit])
        return index

def get_dataset_names(hdf5_file : str) -> dict:
    datasets = {}
    
    with h5py.File(hdf5_file, 'r') as file:
        for model in file.keys():
            if model == 'mapping':
                continue
            for type in file.get(model):
                source_dataset = f"/{model}/{type}/en_cirrussearch"
                query_dataset  = f"/{model}/{type}/lv_cirrussearch"
                datasets[source_dataset] = query_dataset
    return datasets

def search_in_dataset(path_to_csv : str, index : faiss.IndexHNSWFlat, query_dataset : str, k : int, file : h5py.File, log_file : TextIOWrapper) -> None:

    csv_rows = []

    with open(path_to_csv, 'w', newline='\n', encoding='utf-8') as result_csv:
        csv_writer = csv.DictWriter(result_csv, delimiter=';', fieldnames=csv_header)
        csv_writer.writeheader()

        for i in range(len(file[query_dataset])):
            csv_row = {}
            query_file  = file['mapping'][i][0].decode()

            # https://github.com/facebookresearch/faiss/issues/493
            Distances, Indexes = index.search(np.array([file[query_dataset][i]]), k=k)
            Distances = Distances[0]
            Indexes = Indexes[0]

            search_result = [file['mapping'][Indexes[j]][0].decode() for j in range(len(Indexes))]
            search_distances = [str(Distances[j]) for j in range(len(Distances))]
            search_result = ','.join(search_result)
            search_distances = ','.join(search_distances)

            csv_row['search_result'] = search_result
            csv_row['search_distances'] = search_distances
            csv_row['found'] = (query_file in search_result)
            csv_row['query_file'] = query_file

            csv_rows.append(csv_row)
            if i % chunk_size == 0 and i != 0:
                csv_writer.writerows(csv_rows)
                csv_rows.clear()
                log_file.write(f"Flushed '{chunk_size}' search results to csv file\n")
        if len(csv_rows) != 0:
            csv_writer.writerows(csv_rows)
            log_file.write(f"Flushed '{len(csv_rows)}' search results to csv file\n")

if __name__ == '__main__':
    hdf5_file = path_res + "embeddings.hdf5"
    log_file = path_search_FLAT + "search.log" 
    #log_file = path_search + "search.log"
    kNN = [1, 5, 10]

    datasets = get_dataset_names(hdf5_file)

    with open(log_file, 'a', encoding='utf-8') as log_file:
        #try:
        for dataset in datasets:
            index = build_index_from_hdf(hdf5_file, dataset)
            log_file.write(f"Succefully created index for dataset '{dataset}'\n")
            with h5py.File(hdf5_file, 'r') as file:
                query_dataset = datasets[dataset]
                if not query_dataset in file:
                    raise ValueError(f"Query dataset '{query_dataset}' not in file '{hdf5_file}'")
                zero_rows_indexes = get_np_array_zero_rows(file[query_dataset])
                if len(zero_rows_indexes) != 0 and not ignore_zeroes:
                    raise ValueError(f"Embedding in dataset '{query_dataset}' on index '{zero_rows_indexes[0]}' is not calculated!!!")

                for k in kNN:
                    path_to_csv = os.path.join(path_search, dataset.split('/')[1], dataset.split('/')[2], f"{k}NN.csv")
                    #path_to_csv = os.path.join(path_search_FLAT, dataset.split('/')[1], dataset.split('/')[2], f"{k}NN.csv")
                    start = datetime.now()
                    log_file.write(f"Searching consequently for each element of '{query_dataset}'. k = '{k}'. Output to '{path_to_csv}'. Start time = '{start}'\n")
                    search_in_dataset(path_to_csv, index, query_dataset, k, file, log_file) 
                    log_file.write(f"Done. Execution time = '{datetime.now() - start}'\n")
        #except Exception as e:
            #log_file.write(f"Exception occured during searching: '{str(e)}'\n")


