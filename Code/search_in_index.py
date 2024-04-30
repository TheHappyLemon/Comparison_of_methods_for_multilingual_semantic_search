import h5py
from constants import path_res, path_setup
from utils import get_np_array_zero_rows, get_dict_from_json
import faiss
import numpy as np
import csv

def build_index_from_hdf(file_path : str, dataset : str, ignore_zeroes = False) -> faiss.IndexHNSWFlat:
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
        
        index = faiss.IndexHNSWFlat(file[dataset].shape[1], 64)
        index.add(file[dataset][:limit])
        return index

if __name__ == '__main__':
    hdf5_file = path_res + "embeddings_test.hdf5"
    log_file = path_res + "search.log"
    result_file = path_res + "search_result.csv"
    csv_header = ['model', 'source_data', 'query_data', 'query_file', 'NN', 'found', 'search_result', 'search_distances']
    ignore_zeroes = True

    datasets = {}

    kNN = [1, 5, 10]

    with h5py.File(hdf5_file, 'r') as file:
        for model in file.keys():
            if model == 'mapping':
                continue
            for type in file.get(model):
                source_dataset = f"/{model}/{type}/en_cirrussearch"
                query_dataset  = f"/{model}/{type}/lv_cirrussearch"
                datasets[source_dataset] = query_dataset

    with open(result_file, 'w', newline='\n', encoding='utf-8') as result_csv, open(log_file, 'w', encoding='utf-8') as log:
        csv_writer = csv.DictWriter(result_csv, delimiter=';', fieldnames=csv_header)
        csv_writer.writeheader()
        csv_rows = []
        csv_row = {}

        for dataset in datasets:
            index = build_index_from_hdf(hdf5_file, dataset, ignore_zeroes)
            with h5py.File(hdf5_file, 'r') as file:
                query_dataset = datasets[dataset]
                if not query_dataset in file:
                    raise ValueError(f"Query dataset '{query_dataset}' not in file '{hdf5_file}'")
                zero_rows_indexes = get_np_array_zero_rows(file[query_dataset])
                if len(zero_rows_indexes) != 0 and not ignore_zeroes:
                    raise ValueError(f"Embedding in dataset '{query_dataset}' on index '{zero_rows_indexes[0]}' is not calculated!!!")

                
                print(f"Built index of elements in '{dataset}'. Searching consequently for each element of '{query_dataset}'")            
                for i in range(len(file[query_dataset])):
                    if i > 1:
                        break

                    query_file  = file['mapping'][i][0].decode()

                    if ignore_zeroes and i == zero_rows_indexes[0]:
                        break
                    # https://github.com/facebookresearch/faiss/issues/493
                    for k in kNN:
                        csv_row['model'] = dataset.split('/')[1]
                        csv_row['source_data'] = dataset
                        csv_row['query_data'] = query_dataset
                        csv_row['query_file'] = query_file


                        Distances, Indexes = index.search(np.array([file[query_dataset][i]]), k=k)
                        Distances = Distances[0]
                        Indexes = Indexes[0]

                        search_result = [file['mapping'][Indexes[j]][0].decode() for j in range(len(Indexes))]
                        search_distances = [str(Distances[j]) for j in range(len(Distances))]
                        search_result = ','.join(search_result)
                        search_distances = ','.join(search_distances)

                        csv_row['search_result'] = search_result
                        csv_row['search_distances'] = search_distances
                        csv_row[f'NN'] = k
                        csv_row['found'] = (query_file in search_result)
                        #for j in range(len(sorted(Distances))):
                            #source_file = file['mapping'][Indexes[j]][0].decode()
                            #csv_row['search_result'] = source_file
                            #csv_row[f'{k}NN'] = (query_file == source_file)
                            #print(f"Was looking for LV text '{query_file}'. Found EN text '{source_file}' with index {Indexes[j]} and distance: {Distances[j]}")
                        csv_writer.writerow(csv_row)

