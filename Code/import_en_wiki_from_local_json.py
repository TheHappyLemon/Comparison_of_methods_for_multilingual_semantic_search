from constants import path, path_res, path_en_wiki, chunk_size
import csv
from datetime import datetime
import json

def load_info_from_csv(file_path):
    titles_to_filename = {}
    with open(file_path, 'r', newline='\n', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        # skip csv header
        next(csv_reader)
        for page_info in csv_reader:
            en_title = page_info[3]
            file_name = page_info[1]
            titles_to_filename[en_title] = file_name
    return titles_to_filename

def flush_data(data : list):
    # Writes every page text into separate file
    for page in data:
        with open(path_res + f"en_cirrussearch\\{page['file_name']}", 'w', encoding="utf-8") as txtfile: 
            txtfile.write(page['text'])

def process_wikipedia_dump(file_path, origins : dict, chunk_size : int = 500, log_path : str = None):

    with open(file_path, 'r') as file, open(log_path, 'w', encoding='utf-8') as log_file:
        processed_documents = []
        start = datetime.now()
        log_file.write(f"Started importing at '{start}'\n")

        while True:
            # done
            if len(origins) == 0:
                print('breaking')
                break

            line1 = file.readline().strip()
            if not line1:
                # EOF
                break
            line2 = file.readline().strip()
            
            content  = json.loads(line2)
            page_text  = content.get('text' , '')
            page_title = content.get('title', '')
            
            if not page_title in origins:
                continue
            if page_text == "":
                log_file.write(f"Page with title {page_title} has empty text. Skipping it")
                del origins[page_title]
                continue

            processed_document = {
                'text'     : page_text,
                'lv_title' : page_title,
                'file_name': origins[page_title]
            }
            
            processed_documents.append(processed_document)
            del origins[page_title]
            
            if len(origins) % chunk_size == 0:
                flush_data(processed_documents)
                log_file.write(f"flushed '{chunk_size}' documents.'{len(origins)} left'\n")
                processed_documents.clear()
        
        flush_data(processed_documents)
        end = datetime.now()
        log_file.write(f"finished importing at '{end}'\n")
        log_file.write(f"Time elapsed: '{end - start}'\n")


file_path = path_en_wiki + 'enwiki-20240401-cirrussearch-content.json'
log_path  = path_res + 'cirrussearch-content_import_en.log'
origins = load_info_from_csv(path_res + 'results_cirrussearch.csv')
process_wikipedia_dump(file_path, origins, chunk_size, log_path)
