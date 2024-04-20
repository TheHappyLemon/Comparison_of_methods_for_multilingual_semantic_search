import os
import csv
from constants import path_res

def get_filenames(path : str) -> set:
    names = set()   
    filenames = os.listdir(path)

    for filename in filenames:
        full_path = os.path.join(path, filename)
        if os.path.isfile(full_path):
            names.add(filename)
        
    return names

def form_csv_of_common_pages(src : str, res : str, common : set) -> None:
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
                    del line[None]
                full_info.append(line)

    # Write data of common pages to new csv file
    with open(res, 'w', newline='\n', encoding='utf-8') as result:
        csv_writer = csv.DictWriter(result, delimiter=';', fieldnames=header)
        csv_writer.writeheader()
        csv_writer.writerows(full_info)
    

lv_files = get_filenames(path_res + "lv_cirrussearch_title")
en_files = get_filenames(path_res + "en_cirrussearch_title")
common_pages = lv_files.intersection(en_files)
form_csv_of_common_pages(path_res + "results_cirrussearch_ALL.csv", path_res + "results_cirrussearch_ALL_common.csv", common_pages)

