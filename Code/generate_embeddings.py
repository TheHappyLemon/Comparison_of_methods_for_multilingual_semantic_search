
from transformers import BertTokenizer, BertModel
from utils import *
import faiss
from constants import path_res
import os
from exceptions import EmbeddingsError

k = 5

def get_pages_text_count(dataset_path : str, limit : int = -1):
    processed = 0
    if limit != -1:
        return limit

    for filename in os.listdir(dataset_path):
        full_path = os.path.join(dataset_path, filename)
        if os.path.isfile(full_path):
            processed = processed + 1
    return processed

def get_pages_texts(dataset_path : str, start_at : int = 0, stop_at : int = -1, max_limit : int = -1):
    texts = []
    processed = 0
    
    # So order is always the same!
    filenames = sorted(os.listdir(dataset_path))
    if stop_at == -1:
        stop_at = len(filenames)

    for filename in filenames:
        full_path = os.path.join(dataset_path, filename)
        if os.path.isfile(full_path):
            processed = processed + 1
            if processed < start_at or processed > stop_at:
                continue
            with open(full_path, 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
                if max_limit != -1 and len(texts) >= max_limit:
                    break
    return texts

def get_np_array_zero_rows(np_array):
    return np.where(~np.any(np_array, axis=1))[0]

def build_index_from_hdf(file_path : str, dataset : str):
    with h5py.File(file_path, 'r') as file:
        if not dataset in file:
            raise ValueError(f"Dataset {dataset} not in file")
        # np.any - non zero rows, ~ - invert, np.where - return array of indexes with True values
        zero_rows_indexes = get_np_array_zero_rows(file[dataset])
        if len(zero_rows_indexes) != 0:
            raise EmbeddingsError(f"Some embeddings in dataset '{dataset}' are not calculated", zero_rows_indexes[0])
        index = faiss.IndexHNSWFlat(file[dataset].shape[1], 64)
        index.add(file[dataset])
        return index

def get_embedding_from_hdf(file_path : str, dataset : str, index : int):
    with h5py.File(file_path, 'r') as file:
        if not dataset in file:
            raise ValueError(f"Dataset {dataset} not in file")
        return file[dataset][index]
        
if __name__ == '__main__':
    
    index = None
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    hdf5_file = path_res + f'\\embeddings.hdf5'
    en_embeddings_name = "en_embeddings"
    lv_embeddings_name = "lv_embeddings"
    
    try:
        index = build_index_from_hdf(hdf5_file, en_embeddings_name)
    except FileNotFoundError as e1:
        # Generate file with english embeddings from scratch. Will take a lot of time...
        en_texts = get_pages_texts(dataset_path=path_res + "en_cirrussearch_title", max_limit=30)
        generate_embeddings(en_texts,  tokenizer, model, en_embeddings_name)
        #index = build_index_from_hdf(hdf5_file, en_embeddings_name)
    except EmbeddingsError as e2:
        # Some part of embeddings has already been calculated, but some not -> continue
        en_texts = get_pages_texts(dataset_path=path_res + "en_cirrussearch_title", start_at=e2.indexes, max_limit=6)
        generate_embeddings(en_texts, tokenizer, model, en_embeddings_name, batch_offset=e2.indexes)
        #index = build_index_from_hdf(hdf5_file, en_embeddings_name)
    except Exception as e3:
        print("Error:", repr(e3))
        exit()

    with h5py.File(hdf5_file, 'r') as file:
        lv_embeddings_exist = lv_embeddings_name in file
        if lv_embeddings_exist:
            zero_rows_index = get_np_array_zero_rows(file[lv_embeddings_name])[0]
    
    if not lv_embeddings_exist:
        # calculate latvian embeddings if there are none
        lv_texts = get_pages_texts(dataset_path=path_res + "lv_cirrussearch_title", max_limit=5)
        generate_embeddings(lv_texts,  tokenizer, model, lv_embeddings_name)
    else:
        # Some part of embeddings has already been calculated, but some not -> continue
        lv_texts = get_pages_texts(dataset_path=path_res + "lv_cirrussearch_title", start_at=zero_rows_index, max_limit=6)
        generate_embeddings(lv_texts,  tokenizer, model, lv_embeddings_name, batch_offset=zero_rows_index)

    text_count = get_pages_text_count(path_res + "lv_cirrussearch_title", 5)
    for i in range(text_count):
        e = get_embedding_from_hdf(hdf5_file, lv_embeddings_name, i)
        print(type(e))