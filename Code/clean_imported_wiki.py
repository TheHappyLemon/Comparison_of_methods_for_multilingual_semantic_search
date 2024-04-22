import os
import csv
from constants import path_res
from datetime import datetime
from io import TextIOWrapper

def get_filenames(path : str, log : TextIOWrapper) -> set:
    names = set()   
    filenames = os.listdir(path)

    for filename in filenames:
        full_path = os.path.join(path, filename)
        if os.path.isfile(full_path):
            names.add(filename)
        else:
            log.write(f"{full_path} is not a file. Skipping\n")
        
    return names

def form_csv_of_common_pages(src : str, res : str, common : set, log : TextIOWrapper) -> None:
    full_info = []
    header = None

    # Get full data of common pages from csv file
    with open(src, 'r', newline='\n', encoding='utf-8') as source:
        csv_reader = csv.DictReader(source, delimiter=';')
        header = csv_reader.fieldnames
        for line in csv_reader:
            if line['file_name'] in common:
                if line['id'] == '40228':
                    # One page had ';' in title and I did not quote it when creating csv file, so fix it manually here
                    line['lv_title'] = line['lv_title'] + ";" + line['en_title']
                    line['en_title'] = line['lv_title']
                    line['lv_link']  = f"https://lv.wikipedia.org/wiki/{line['lv_title']}"
                    line['en_link']  = f"https://en.wikipedia.org/wiki/{line['lv_title']}"
                    log.write(f"Fixed entry with id '{line['id']}', because did not quote it when creating csv file\n")
                    del line[None]
                full_info.append(line)
            else:
                log.write(f"Skipped '{line['id']}' - '{line['lv_title']}' because it is not in english wiki dataset\n")

    # Write data of common pages to new csv file
    with open(res, 'w', newline='\n', encoding='utf-8') as result:
        csv_writer = csv.DictWriter(result, delimiter=';', fieldnames=header)
        csv_writer.writeheader()
        csv_writer.writerows(full_info)
    log.write(f"Result containts '{len(full_info)}' rows\n")
    
def delete_uncommon_pages_from(path : str, uncommon : set, log : TextIOWrapper) -> None:
    filenames = os.listdir(path)

    for filename in filenames:
        full_path = os.path.join(path, filename)
        if os.path.isfile(full_path) and filename in uncommon:
            try:
                os.remove(full_path)
            except Exception as e:
                log.write(f"Failed to delete file '{full_path}', error: {e}\n")

with open(path_res + 'dataset_cleaning.log', 'w', encoding='utf-8') as log:
    start = datetime.now()
    log.write("Gathering filenames for lv and en pages...\n")
    lv_files = get_filenames(path_res + "lv_cirrussearch_title", log)
    en_files = get_filenames(path_res + "en_cirrussearch_title", log)
    log.write(f"Gathered filenames! '{len(lv_files)}' lv pages and '{len(en_files)}' pages\n")

    common_pages = lv_files.intersection(en_files)
    uncommon_pages = lv_files.difference(en_files)
    log.write(f"According to filenames, there is '{len(common_pages)}' common pages and '{len(uncommon_pages)}' uncommon pages!\n")
    
    form_csv_of_common_pages(path_res + "results_cirrussearch_ALL.csv", path_res + "results_cirrussearch_ALL_common.csv", common_pages, log)
    log.write(f"Formed csv of common pages - 'results_cirrussearch_ALL.csv'\n")
    delete_uncommon_pages_from(path_res + "lv_cirrussearch_title" , uncommon_pages, log)
    delete_uncommon_pages_from(path_res + "lv_cirrussearch_open"  , uncommon_pages, log)
    delete_uncommon_pages_from(path_res + "lv_cirrussearch_source", uncommon_pages, log)
    log.write(f"Deleted uncommong files from lv wiki directories!\n")
    log.write(f"Working time = '{datetime.now() - start}'\n")

