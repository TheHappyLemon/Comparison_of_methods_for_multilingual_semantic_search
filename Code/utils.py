import torch
import requests
import numpy as np
import json
import os

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
            if processed <= start_at or processed > stop_at:
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