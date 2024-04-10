import torch
import requests

def cls_pooling(model_output):
    # take embedding of first token of every sequence (CLS) token, which is a special classification token 
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list : list, tokenizer, model, truncation : bool = False, max_length : int = 512):
    # calls tokenizer associated with provided model
    # which procceses input, so model is ready to work with it
    # padding = extend the sequences to the length of the longest sequence or max_length if provided
    # truncation = truncate longest sentences to max_length
    # max_length = maximum sequence length
    encoded_input = tokenizer(text_list, padding=True, return_tensors="pt", truncation=truncation, max_length=max_length)
    # torch.no_grad - we are NOT training. It does not store gradients during forward pass
    with torch.no_grad():
        model_output = model(**encoded_input)
    return cls_pooling(model_output)

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