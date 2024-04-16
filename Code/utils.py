import torch
import requests
import numpy as np
import h5py
from constants import path_res

def cls_pooling(model_output):
    # take embedding of first token of every sequence (CLS) token, which is a special classification token
    return model_output.last_hidden_state[:, 0]

def flush_embeddings(embeddings_list : list, group_name : str, cur_batch : int, total_vectors : int, emb_dimensions : int) -> None:
    with h5py.File(path_res + f'\\embeddings.hdf5', 'a') as file:
        if not group_name in file:
            file.create_dataset(group_name, shape=(total_vectors, emb_dimensions))
        for i in range(len(embeddings_list)):
            embeddings_vector = embeddings_list[i]
            file[group_name][cur_batch + i] = embeddings_vector

def generate_embeddings(text_list, tokenizer, model, file_name : str, batch_size=10, log_file = None, ):
    
    # calls tokenizer associated with provided model
    # which procceses input, so model is ready to work with it
    # padding = extend the sequences to the length of the longest sequence or max_length if provided
    # truncation = truncate longest sentences to max_length
    # max_length = maximum sequence length
    # https://huggingface.co/docs/transformers/pad_truncation padding to max model input length

    # Process texts in batches and flush each batch to the disk
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]
        encoded_input = tokenizer(batch_texts, padding='max_length', truncation=True, return_tensors="pt")
        
        # torch.no_grad - we are NOT training. It does not store gradients during forward pass
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = cls_pooling(model_output).numpy()

        flush_embeddings(embeddings, file_name, i, len(text_list), embeddings.shape[1])
        if log_file != None:
            log_file.write(f"Flushed '{i + batch_size}' texts embeddings\n")
        else:
            print(f"Flushed '{i + batch_size}' units")

    return embeddings.shape[1]

def get_wiki_data(title : str, log_file = None):

    # accepts title and log_file object (optional) and makes request to wikipedia to retrieve all
    # analogies of this page in other languages
    # returns list of three elements: en_url, lv_url, en_title
    # if errors occur they are outputed to file (if provided) or to default stream

    response = requests.get(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbgetentities",
            "titles": title,
            "sites": "lvwiki",
            "props": "sitelinks/urls",
            "format": "json"
        }
    )
    if response.status_code != 200:
        return None
    response = response.json()

    try:
        sitelinks = response['entities'][next(iter(response['entities']))]['sitelinks']
        en_link  = sitelinks['enwiki']['url']
        lv_link  = sitelinks['lvwiki']['url']
        en_title = sitelinks['enwiki']['title']
        return [en_link, lv_link, en_title]
    except KeyError as e:
        err_txt = f"error '{repr(e)}' for title '{title}'"
        if log_file == None:
            print(err_txt)
        else:
            log_file.write(err_txt + '\n')
        return None