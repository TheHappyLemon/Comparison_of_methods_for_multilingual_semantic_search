
from transformers import BertTokenizer, BertModel
from utils import *
import faiss
from constants import path_res
import os

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

def get_pages_texts(dataset_path : str, max_limit : int = -1):
    texts = []
    processed = 0
    
    for filename in os.listdir(dataset_path):
        full_path = os.path.join(dataset_path, filename)
        if os.path.isfile(full_path):
            with open(full_path, 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
                processed = processed + 1
                if max_limit != -1 and processed >= max_limit:
                    break
    return texts

def build_index_from_hdf(file_path : str, dataset : str):
    with h5py.File(file_path, 'r') as file:
        if not dataset in file:
            raise ValueError(f"Dataset {dataset} not in file")
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
    
    #TODO
    # Check if dataset {lv/en}_embeddings has as many entries as wikipedia texts are stored on a disk 

    try:
        index = build_index_from_hdf(hdf5_file, en_embeddings_name)
    except (FileNotFoundError, ValueError) as e:
        # Generate file with english embeddings. Will take a lot of time...
        en_texts = get_pages_texts(path_res + "en_cirrussearch_title", 30)
        generate_embeddings(en_texts,  tokenizer, model, en_embeddings_name)
        index = build_index_from_hdf(hdf5_file, en_embeddings_name)
    except Exception as e:
        print("Error:", repr(e))
        exit()

    try:
        get_embedding_from_hdf(hdf5_file, lv_embeddings_name, 0)
        text_count = get_pages_text_count(path_res + "lv_cirrussearch_title", 5)
    except (ValueError, IndexError) as e:
        lv_texts = get_pages_texts(path_res + "lv_cirrussearch_title", 5)
        generate_embeddings(lv_texts,  tokenizer, model, lv_embeddings_name)
        text_count = len(lv_texts)
    
    for i in range(text_count):
        e = get_embedding_from_hdf(hdf5_file, lv_embeddings_name, i)
        print(type(e))



    


    # lv_texts = get_pages_texts(path_res + "en_cirrussearch_title", 5)
    # lv_embeddings = get_embeddings(lv_texts, tokenizer, model)
    # for lv_embedding in lv_embeddings:
    #     lv_embedding_2d = np.expand_dims(lv_embedding, axis=0)
    #     D, I = index.search(lv_embedding_2d, k)
    #     for i in I[0]:  
    #         print(f"Index {i} text: {en_texts[i]}")
        