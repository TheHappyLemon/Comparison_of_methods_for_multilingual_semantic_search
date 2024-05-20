from laser_encoders import initialize_encoder, initialize_tokenizer
from constants import path_res, path_log
from os.path import sep

path = path_res + f'en_cirrussearch_source{sep}000000.txt' 
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = initialize_tokenizer(lang="eng_Latn")
encoder = initialize_encoder(lang="eng_Latn")

################## from laser_tokenizer.py ######################################
# encoded_text = " ".join(self.spm_encoder.encode(sentence_text, out_type=str)) #
#################################################################################

with open(path_log + 'test_laser_tokenizer.log', 'w') as log:
    tokenized_sentence = tokenizer.tokenize(text)
    log.write(f"Text was tokenized into '{len(tokenized_sentence.split(' '))}' tokens.\n")
    embeddings = encoder.encode_sentences([tokenized_sentence])
    log.write(f"Embeddings shape is '{embeddings[0].shape[0]}'")