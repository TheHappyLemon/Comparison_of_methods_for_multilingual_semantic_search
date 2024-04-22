import torch
import requests
import numpy as np
import h5py
from constants import path_res

def cls_pooling(model_output):
    # take embedding of first token of every sequence (CLS) token, which is a special classification token
    return model_output.last_hidden_state[:, 0]

def get_embedding(text, tokenizer, model):
    encoded_input = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=model.config.max_position_embeddings)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = cls_pooling(model_output).numpy()
    return embedding

def get_np_array_zero_rows(np_array):
    return np.where(~np.any(np_array, axis=1))[0]

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