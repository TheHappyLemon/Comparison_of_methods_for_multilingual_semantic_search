from laser_encoders import LaserEncoderPipeline
from datetime import datetime
from constants import path_res, path_setup
from utils import get_dict_from_json, get_pages_data, get_pages_text_count, get_np_array_zero_rows
from io import TextIOWrapper
import h5py

encoder_en = LaserEncoderPipeline(lang="eng_Latn")
encoder_lv = LaserEncoderPipeline(lang="lvs_Latn")

def create_datasets_if_not_exist(model_name : str, file_name : str, wiki_types : list, embedding_langs : list, log : TextIOWrapper):

    pages_count = -1
    filenames = []

    with h5py.File(file_name, 'a') as file:
        if not model_name in file:
            test_case = encoder_en.encode_sentences(["Test"])
            model_group = file.create_group(model_name)
            for wiki_type in wiki_types:
                type_group = model_group.create_group(wiki_type)
                for lang in embedding_langs:
                    if pages_count == -1:
                        pages_count = get_pages_text_count(path_res + lang + '_' + wiki_type)
                    if (filenames == []) and (not 'mapping' in file):
                        filenames = get_pages_data(path_res + lang + '_' + wiki_type, type='names')
                    #type_group.create_dataset(lang, shape=(cache[path_to_data], test_case.shape[1]), compression='gzip', chunks=(5, test_case.shape[1]))
                    r = type_group.create_dataset(lang, shape=(pages_count, test_case.shape[1]), chunks=(25, test_case.shape[1]))
                    log.write(f"Created dataset '{r.name}' with shape = '{(pages_count, test_case.shape[1])}'\n")
        
        if not 'mapping' in file:
            r = file.create_dataset('mapping', shape=(pages_count, 1), chunks=(25, 1), data=filenames)
            log.write(f"Created dataset '{r.name}' to map dataset indexes with filenames\n")


def fill_datasets_if_empty(model_name : str, file_name : str, wiki_types : list, embedding_langs : list, log : TextIOWrapper):

    batch_size = 25
    max_limit = 10

    with h5py.File(file_name, 'a') as file:
        group_name = file[model_name]
        for wiki_type in wiki_types:
            type_group = group_name[wiki_type]
            for lang in embedding_langs:
                dataset = type_group[lang]
                zero_rows_index = get_np_array_zero_rows(dataset)
                # Start recalculating from first zero row. Rows are calculated one by one, so rows after also must be zeros
                if len(zero_rows_index) != 0:
                    batch_offset = zero_rows_index[0]
                    texts = get_pages_data(path_res + lang + '_' + wiki_type, start_at=batch_offset, max_limit=max_limit, type='texts')

                    log.write(f"Calculating embeddings for {dataset.name}. Starting from index {batch_offset}\n")

                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i + batch_size]
                        if lang.startswith("lv"):
                            embeddings = encoder_lv.encode_sentences(batch_texts)
                        elif lang.startswith("en"):
                            embeddings = encoder_en.encode_sentences(batch_texts)
                        else:
                            raise ValueError("Languge unsupported")

                        for j in range(len(embeddings)):
                            dataset[i + j + batch_offset] = embeddings[j]
                        if i != 0 and i % 1000 == 0 :
                            log.write(f"Processed a thousand records!\n")
                        file.flush()
                        log.flush()

if __name__ == '__main__':
    log_path = path_res + "embedding_generation_LASER.log"
    hdf5_file = path_res + f'embeddings.hdf5'
    wiki_types = get_dict_from_json(path_setup + "wiki_types.json")
    embedding_langs = get_dict_from_json(path_setup + "embedding_types.json")
    if (wiki_types is None) or (embedding_langs is None):
        print(f"Some setup files are not on disk at '{path_setup}'!")
        exit()

    wiki_types = [key for key, value in wiki_types.items() if value is True]
    embedding_langs = [key for key, value in embedding_langs.items() if value is True]
    
    with open(log_path, 'a', encoding='utf-8') as log:
    # Create file with precalculated embeddings
        start = datetime.now()
        log.write("Start working on filling LASER dataset in hdf5 file\n")
        create_datasets_if_not_exist("LASER", hdf5_file, wiki_types, embedding_langs, log)
        fill_datasets_if_empty("LASER", hdf5_file, wiki_types, embedding_langs, log)
        log.write("Finished!\n")
        log.write(f"Time of execution = {datetime.now() - start}\n")