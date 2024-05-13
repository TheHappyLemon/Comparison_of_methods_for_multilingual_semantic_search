import sys
from laser_encoders import LaserEncoderPipeline
from transformers import AutoTokenizer, AutoModel
from utils import *
from constants_hpc import *
from datetime import datetime

def main():
    if len(sys.argv) != 5:
        print("Usage: python print_arg.py <model> <type> <lang> <size>")
        return
    batch_size = 25

    model_short = sys.argv[1].lower()
    type  = sys.argv[2]
    lang  = sys.argv[3]
    size  = int(sys.argv[4])
    
    print()
    print('************************************************ START *************************************************************')
    print(f"Measuring time with parameters: model = '{model_short}', type = '{type}', lang = '{lang}', texts_amount = '{size}'")

    path = os.path.join(path_res, f"{lang}_cirrussearch_{type}") 
    print('path for texts =', path)

    texts = get_pages_data(path, max_limit=size, type='texts')
    print(f'Gathered "{len(texts)}" texts for testing')
    if model_short != 'laser':
        if model_short == 'bert':
            full_model = 'bert-base-multilingual-cased'
        elif model_short == 'roberta':
            full_model = 'xlm-roberta-base'
        tokenizer = AutoTokenizer.from_pretrained(full_model)
        model = AutoModel.from_pretrained(full_model)

        start = datetime.now()
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded_input = tokenizer(batch_texts, padding='max_length', truncation=True, return_tensors="pt")
            
            with torch.no_grad():
                model_output = model(**encoded_input)
            embeddings = cls_pooling(model_output).numpy()
        total_time = datetime.now() - start
        print(f"Time of execution = '{total_time}'")
            
    else:
        encoder_en = LaserEncoderPipeline(lang="eng_Latn")
        encoder_lv = LaserEncoderPipeline(lang="lvs_Latn")
        if lang == "lv":
            start = datetime.now()
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                embeddings = encoder_lv.encode_sentences(batch_texts)
            total_time = datetime.now() - start
        elif lang == 'en':
            start = datetime.now()
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                embeddings = encoder_en.encode_sentences(batch_texts)
            total_time = datetime.now() - start
        print(f"Time of execution = '{total_time}'")


    print()
    print('************************************************ END *************************************************************')

if __name__ == "__main__":
    main()