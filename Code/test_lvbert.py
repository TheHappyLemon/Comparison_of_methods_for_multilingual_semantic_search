from transformers import AutoTokenizer, AutoModel
from utils import get_embedding
import faiss
import numpy as np

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("AiLab-IMCS-UL/lvbert")
model = AutoModel.from_pretrained("AiLab-IMCS-UL/lvbert")
texts = ["Es staigāju mežā", "dzīvnieks ir liels", "Es domāju"]

#encoded_input = tokenizer(text, return_tensors='pt')
# {'input_ids': tensor([[   2,  181, 7491,  125, 4217,    3]]),
# These are token ids from model vocabulary - https://huggingface.co/AiLab-IMCS-UL/lvbert/raw/main/vocab.txt
# for  example 181 = Es, 7491 = staigā, 125 = ju, 4217 = mežā


if __name__ == '__main__':
    embeddings = get_embedding(texts, tokenizer, model)
    dimension = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dimension, 64)
    index.add(embeddings.astype(np.float32))

    query_embedding = get_embedding(["Es atpušos mežā"], tokenizer, model)
    k = 3
    D, I = index.search(query_embedding[0:1], k)
    print("Distances:", D)
    print("Indices:", I)
    for i in I[0]:  
        print(f"Index {i} text: {texts[i]}")