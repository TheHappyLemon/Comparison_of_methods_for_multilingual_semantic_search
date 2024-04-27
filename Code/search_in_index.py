import h5py
from constants import path_res
from utils import get_np_array_zero_rows
import faiss
import numpy as np

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

def get_embedding_from_hdf(file_path : str, dataset : str, index : int):
    pass

if __name__ == '__main__':
    hdf5_file = path_res + "embeddings_test.hdf5"
    ignore_zeroes = True

    source_dataset = "/LASER/open/en_cirrussearch"
    query_dataset  = "/LASER/open/lv_cirrussearch"
    kNN = [1, 5, 10]

    index = build_index_from_hdf(hdf5_file, source_dataset, ignore_zeroes)
    with h5py.File(hdf5_file, 'r') as file:
        if not query_dataset in file:
            raise ValueError(f"Dataset '{query_dataset}' not in file")
        zero_rows_indexes = get_np_array_zero_rows(file[query_dataset])
        if len(zero_rows_indexes) != 0 and not ignore_zeroes:
            raise ValueError(f"Embedding in dataset '{query_dataset}' on index '{zero_rows_indexes[0]}' is not calculated!!!")
        
        for i in range(len(file[query_dataset])):
            if ignore_zeroes and i == zero_rows_indexes[0]:
                break
            # https://github.com/facebookresearch/faiss/issues/493
            for k in kNN:
                Distances, Indexes = index.search(np.array([file[query_dataset][i]]), k=k)
                Distances = Distances[0]
                Indexes = Indexes[0]
                for j in range(len(sorted(Distances))):
                    query_file  = file['mapping'][i][0].decode()
                    source_file = file['mapping'][Indexes[j]][0].decode()
                    print(f"Was looking for LV text '{query_file}'. Found EN text '{source_file}' with index {Indexes[j]} and distance: {Distances[j]}")
