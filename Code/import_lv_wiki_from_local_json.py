import json
from constants import path, path_res, chunk_size
from utils import get_wiki_data
from datetime import datetime

def flush_data(data : list, has_flushed : bool):
    # writes pages meta data to csv file
    # and then writes every page text into separate file
    with open(path_res + 'results_cirrussearch_15k.csv', 'a', encoding="utf-8") as csvfile:
        if not has_flushed:
            csvfile.write('id;file_name;lv_title;en_title;lv_link;en_link\n')
        for page in data:
            csvfile.write(f"{page['id']};{page['file_name']};{page['lv_title']};{page['en_title']};{page['lv_url']};{page['en_url']}\n")
            with open(path_res + f"lv_cirrussearch_source\\{page['file_name']}", 'w', encoding="utf-8") as txtsrc: 
                txtsrc.write(page['text'])
                txtsrc.flush()
                txtsrc.close()
            with open(path_res + f"lv_cirrussearch_open\\{page['file_name']}", 'w', encoding="utf-8") as txtopn: 
                txtopn.write(page['open_text'])
                txtopn.flush()
                txtopn.close()
            with open(path_res + f"lv_cirrussearch_title\\{page['file_name']}", 'w', encoding="utf-8") as txtttl: 
                txtttl.write(page['lv_title'])
                txtttl.flush()
                txtttl.close()
        csvfile.flush()
        csvfile.close()

def process_wikipedia_dump(file_path, chunk_size : int = 500, max_pages = 10, log_path : str = None):

    processed_total = 0
    has_flushed = False

    with open(file_path, 'r') as file, open(log_path, 'w', encoding='utf-8') as log_file:
        processed_documents = []
        start = datetime.now()
        log_file.write(f"Started importing at '{start}'\n")

        while True:

            # debugging
            if processed_total >= max_pages:
                break

            line1 = file.readline().strip()
            if not line1:
                # EOF
                break
            line2 = file.readline().strip()
            
            metadata = json.loads(line1)
            content  = json.loads(line2)
            
            page_id    = metadata['index']['_id']
            page_text  = content.get('text' , '')
            page_open_text = content.get('opening_text', '')
            page_title = content.get('title', '')
            
            # page_open_text is None - because some articles for some reason do not have property opening_text...
            if page_title == "" or page_text == "" or (page_open_text == "" or page_open_text is None):
                continue

            data = get_wiki_data(page_title, log_file)
            if data == None:
                continue

            f_name = str(processed_total).zfill(6) + '.txt'

            processed_document = {
                'id'       : processed_total,
                'file_name': f_name,
                'lvwiki_id': page_id,
                'text'     : page_text,
                'open_text': page_open_text,
                'lv_title' : page_title,
                'en_title' : data[2],
                'lv_url'   : data[1],
                'en_url'   : data[0],
            }
            
            processed_documents.append(processed_document)
            processed_total = processed_total + 1
            
            if processed_total % chunk_size == 0:
                flush_data(processed_documents, has_flushed)
                has_flushed = True
                log_file.write(f"flushed '{chunk_size}' documents. Currently at '{processed_total}'\n")
                processed_documents.clear()
        
        flush_data(processed_documents, has_flushed)
        end = datetime.now()
        log_file.write(f"finished importing at '{end}'\n")
        log_file.write(f"Time elapsed: '{end - start}'\n")

file_path = path + 'lvwiki-20240401-cirrussearch-content.json'
log_path  = path_res + 'cirrussearch-content_import_lv_15k.log'
max_pages = 15000
process_wikipedia_dump(file_path, chunk_size, max_pages, log_path)