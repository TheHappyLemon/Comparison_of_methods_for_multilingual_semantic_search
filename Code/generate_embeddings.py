
from transformers import AutoTokenizer, AutoModel
from utils import *
import faiss
from constants import path_res, path_setup
import os
from exceptions import EmbeddingsError
import json
from datetime import datetime

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

def get_pages_data(dataset_path : str, start_at : int = 0, stop_at : int = -1, max_limit : int = -1, type='texts'):
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
                if type == 'texts':
                    text = file.read()
                    texts.append(text)
                else:
                    texts.append(filename)
                if max_limit != -1 and len(texts) >= max_limit:
                    break
    return texts
        
def get_dict_from_json(file : str):
    try:
        with open(file, 'r') as f:
            res = json.load(f)
            return res
    except FileNotFoundError:
        print(file)
        return None

def create_datasets_if_not_exist(file_name : str, names : dict, wiki_types : list, embedding_langs : list):

    ###### FILE STRUCTURE ######
    #    Model_name            #
    #    ├── title             #
    #    │   ├── en            #
    #    │   └── lv            #
    #    ├── open              #
    #    │   ├── en            #
    #    │   └── lv            #
    #    └── source            #
    #    │   ├── en            #
    #    │   └── lv            #
    #    ...                   #
    ############################

    pages_count = -1
    filenames = []

    with h5py.File(file_name, 'a') as file:
        for model_name in names:
            tokenizer = AutoTokenizer.from_pretrained(names[model_name])
            model = AutoModel.from_pretrained(names[model_name])
            if not model_name in file:
                test_case = get_embedding("Test", tokenizer, model)
                model_group = file.create_group(model_name)
                for wiki_type in wiki_types:
                    type_group = model_group.create_group(wiki_type)
                    for lang in embedding_langs:
                        if pages_count == -1:
                            pages_count = get_pages_text_count(path_res + lang + '_' + wiki_type)
                        if filenames == []:
                            filenames = get_pages_data(path_res + lang + '_' + wiki_type, type='names')
                        #type_group.create_dataset(lang, shape=(cache[path_to_data], test_case.shape[1]), compression='gzip', chunks=(5, 768))
                        type_group.create_dataset(lang, shape=(pages_count, test_case.shape[1]), chunks=(25, 768))
        
        if not 'mapping' in file:
            file.create_dataset('mapping', shape=(pages_count, 1), chunks=(25, 1), data=filenames)

def fill_datasets_if_empty(file_name : str, names : dict, wiki_types : list, embedding_langs : list):

    batch_size = 10
    start = datetime.now()

    with h5py.File(file_name, 'a') as file:
        for model_name in names:

            tokenizer = AutoTokenizer.from_pretrained(names[model_name])
            model = AutoModel.from_pretrained(names[model_name])

            group_name = file[model_name]
            for wiki_type in wiki_types:
                type_group = group_name[wiki_type]
                for lang in embedding_langs:
                    dataset = type_group[lang]
                    zero_rows_index = get_np_array_zero_rows(dataset)
                    # Start recalculating from first zero row. Rows are calculated one by one, so rows after also must be zeros
                    if len(zero_rows_index) != 0:
                        batch_offset = zero_rows_index[0]
                        texts = get_pages_data(path_res + lang + '_' + wiki_type, start_at=batch_offset, max_limit=25, type='texts')

                        for i in range(0, len(texts), batch_size):
                            batch_texts = texts[i:i + batch_size]
                            encoded_input = tokenizer(batch_texts, padding='max_length', truncation=True, return_tensors="pt")
                            
                            # torch.no_grad - we are NOT training. It does not store gradients during forward pass
                            with torch.no_grad():
                                model_output = model(**encoded_input)
                            embeddings = cls_pooling(model_output).numpy()

                            for j in range(len(embeddings)):
                                dataset[i + j + batch_offset] = embeddings[j]
                            print(f"Flushed {len(batch_texts)} units")
                            file.flush()
    end = datetime.now()
    print(f'Been working {end - start}')

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
    
    # Create file with precalculated embeddings
    create_datasets_if_not_exist(hdf5_file, models, wiki_types, embedding_langs)
    fill_datasets_if_empty(hdf5_file, models, wiki_types, embedding_langs)