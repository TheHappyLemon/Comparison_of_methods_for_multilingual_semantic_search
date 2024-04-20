
from transformers import AutoTokenizer, AutoModel
from utils import *
import faiss
from constants import path_res, path_setup
import os
from exceptions import EmbeddingsError
import json
from pathlib import Path

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
        
def get_dict_from_json(file : str):
    try:
        with open(file, 'r') as f:
            res = json.load(f)
            return res
    except FileNotFoundError:
        print(file)
        return None

def fill_datasets_if_empty(file_name : str, names : dict, wiki_types : list, embedding_langs : list):

    batch_size = 10

    path = ''
    sep  = '\\'
    #  Path("/my/directory").mkdir(parents=True, exist_ok=True)
    for model_name in names:

        tokenizer = AutoTokenizer.from_pretrained(names[model_name])
        model = AutoModel.from_pretrained(names[model_name])

        path = model_name

        for wiki_type in wiki_types:
            path = path + sep + wiki_type

            for lang in embedding_langs:
                path = path + sep + lang
                
                zero_rows_index = get_np_array_zero_rows(dataset)
                # Start recalculating from first zero row. Rows are calculated one by one, so rows after also must be zeros
                if len(zero_rows_index) != 0:
                    batch_offset = zero_rows_index[0]
                    texts = get_pages_texts(path_res + lang + '_' + wiki_type, start_at=batch_offset, max_limit=15)

                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i + batch_size]
                        encoded_input = tokenizer(batch_texts, padding='max_length', truncation=True, return_tensors="pt")
                        
                        # torch.no_grad - we are NOT training. It does not store gradients during forward pass
                        with torch.no_grad():
                            model_output = model(**encoded_input)
                        embeddings = cls_pooling(model_output).numpy()

                        for j in range(len(embeddings)):
                            dataset[i + j + batch_offset] = embeddings[j]
                        #file.flush()


if __name__ == '__main__':
    # definitions
    hdf5_file = path_res + f'\\embeddings.hdf5'

    # Read setup files and extract which models we want, what kind of wikipedia pages and what language embeddings to generate
    models = get_dict_from_json(path_setup + "models.json")
    wiki_types = get_dict_from_json(path_setup + "wiki_types.json")
    embedding_langs = get_dict_from_json(path_setup + "embedding_types.json")
    if (models is None) or (wiki_types is None) or (embedding_langs is None):
        print(f"Some setup files are not on disk at '{path_setup}'!")
        exit()
    wiki_types = [key for key, value in wiki_types.items() if value is True]
    embedding_langs = [key for key, value in embedding_langs.items() if value is True]
    
    create_datasets_if_not_exist(hdf5_file, models, wiki_types, embedding_langs)
    fill_datasets_if_empty(hdf5_file, models, wiki_types, embedding_langs)