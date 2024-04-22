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
    hdf5_file = path_res + "embeddings.hdf5"
    ignore_zeroes = True

    en_1 = "/XLM-RoBERTa/title/en_cirrussearch"
    lv_1 = "/XLM-RoBERTa/title/lv_cirrussearch"
    k = 5

    #try:
    index_en_open = build_index_from_hdf(hdf5_file, en_1, ignore_zeroes)
    with h5py.File(hdf5_file, 'r') as file:
        if not lv_1 in file:
            raise ValueError(f"Dataset {lv_1} not in file")
        zero_rows_indexes = get_np_array_zero_rows(file[lv_1])
        if len(zero_rows_indexes) != 0 and not ignore_zeroes:
            raise ValueError(f"Embedding in dataset '{lv_1}' on index '{zero_rows_indexes[0]}' is not calculated!!!")
        for i in range(len(file[lv_1])):
            if i == zero_rows_indexes[0]:
                break
            # https://github.com/facebookresearch/faiss/issues/493
            D, I = index_en_open.search(np.array([file[lv_1][i]]), k=k)
            D = D[0]
            I = I[0]
            print(D, I)
            for j in range(len(I)):
                print(f"Was looking for LV text '{file['mapping'][i]}'. Found EN text'{file['mapping'][I[j]]}' with index {I[j]} and distance: {D[j]}")
                
    #except ValueError as e:
       # print(e)
    #except IndexError as e:
        #print(e)