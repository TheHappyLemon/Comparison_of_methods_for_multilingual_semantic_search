
from transformers import BertTokenizer, BertModel
from utils import *
import faiss
import numpy as np
from constants import path_res
import os

k = 5

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


if __name__ == '__main__':
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')

    en_texts = get_pages_texts(path_res + "en_cirrussearch_title", 5)
    en_embeddings = get_embeddings(en_texts, tokenizer, model).numpy()
    dimensions = en_embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dimensions, 64)
    index.add(en_embeddings.astype(np.float32))

    lv_texts = get_pages_texts(path_res + "en_cirrussearch_title", 5)
    lv_embeddings = get_embeddings(lv_texts, tokenizer, model).numpy()
    for lv_embedding in lv_embeddings:
        lv_embedding_2d = np.expand_dims(lv_embedding, axis=0)
        D, I = index.search(lv_embedding_2d, k)
        for i in I[0]:  
            print(f"Index {i} text: {en_texts[i]}")
        