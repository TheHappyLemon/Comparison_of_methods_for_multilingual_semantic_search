from transformers import AutoTokenizer, AutoModel
from utils import *
import faiss
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
texts = ["The apple is tasty", "The weather is bad", "I took my dog for a play"]  # Added another text for demonstration

if __name__ == '__main__':
    embeddings = get_embeddings(texts, tokenizer, model)
    print(embeddings.size())
    embeddings = embeddings.numpy()
    dimension = embeddings.shape[1]

    # index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.add(embeddings.astype(np.float32))
    print(index.is_trained)

    query_embedding = get_embeddings(["I took my cat for a walk"], tokenizer, model).numpy()

    k = 1
    D, I = index.search(query_embedding[0:1], k)
    print("Distances:", D)
    print("Indices:", I)
    for i in I[0]:  
        print(f"Index {i} text: {texts[i]}")