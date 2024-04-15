
from transformers import BertTokenizer, BertModel
from utils import *
import faiss
import numpy as np
from constants import path_res

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

if __name__ == '__main__':
    
    topics = ["Medicina", "Astronomy", "zvaigzne"]
    embeddings = get_embeddings(topics, tokenizer, model).numpy()
    dimensions = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dimensions, 64)
    index.add(embeddings.astype(np.float32))


    with open(path_res + "lv_cirrussearch_title\\000000.txt", 'r', encoding='utf-8') as input_text:
        texts = input_text.readlines()
    embeddings = get_embeddings(texts, tokenizer, model)
    embeddings = embeddings.numpy()

    k = 3
    D, I = index.search(embeddings, k)
    print("Distances:", D)
    print("Indices:", I)
    for i in I[0]:  
        print(f"Index {i} text: {topics[i]}")