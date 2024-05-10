import csv
from constants import path_res, path_search
import h5py 

def write_to_csv():
    with open(path_res + 'same_titles.csv', 'w', newline='\n', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['file_name', 'lv_title', 'en_title'])
        for page in same_titles:
            writer.writerow(page)

def get_pages_with_same_title(file_path):
    res = []
    with open(file_path, 'r', newline='\n', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            if row[2] == 'lv_title':
                continue
            if row[2] == row[3]:
                data = (row[1], row[2], row[3])
                res.append(data)
    return res

def check_if_contents_are_really_the_same(same_titles):
    i = 0
    for same_title in same_titles:
        i = i + 1
        en_contents = None
        lv_contents = None
        with open(path_res + 'en_cirrussearch_title\\' + same_title[0], 'r', encoding='utf-8') as en_file:
            en_contents = en_file.read()
        with open(path_res + 'lv_cirrussearch_title\\' + same_title[0], 'r', encoding='utf-8') as en_file:
            lv_contents = en_file.read()
        if lv_contents != en_contents:
            print('Error found!!!:')
            print(same_title[0], en_contents, lv_contents, en_contents == lv_contents)

def check_if_pages_found_perfectly_in_file(filenames, file_path):
    was_unable_to_find_perfectly = []
    with open(file_path, 'r', newline='\n', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            if row[0] in filenames:
                if row[1] == 'False' or row[3] != '0.0':
                    was_unable_to_find_perfectly.append(row[0])
    return was_unable_to_find_perfectly

same_titles = get_pages_with_same_title( path_res + 'results_cirrussearch_ALL_common.csv')
check_if_contents_are_really_the_same(same_titles)
#write_to_csv()
filenames = [data[0] for data in same_titles]
laser_data = check_if_pages_found_perfectly_in_file(filenames, path_search + 'LASER\\title\\1NN.csv')
robert_data = check_if_pages_found_perfectly_in_file(filenames, path_search + 'XLM-RoBERTa\\title\\1NN.csv')
bert_data = check_if_pages_found_perfectly_in_file(filenames, path_search + 'BERT\\title\\1NN.csv')

print(bert_data)
exit()
with h5py.File(path_res + 'embeddings.hdf5', 'r') as file:
    total = 0
    mapping_set = file['/mapping']
    hdf5_indexes = []
    for i in range(len(mapping_set)):
        data = mapping_set[i][0].decode()
        if data in laser_data:
            hdf5_indexes.append(i)
    print(len(hdf5_indexes), len(laser_data))
    print('Comparing embeddings...')
    not_ok = set()
    for index in hdf5_indexes:
        lv_embd = file['/LASER/title/en_cirrussearch'][index]
        en_embd = file['/LASER/title/lv_cirrussearch'][index]
        for i in range(len(lv_embd)):
            difference = abs(lv_embd[i] - en_embd[i])
            if difference != 0.0:
                print(lv_embd[i], en_embd[i], difference)
                not_ok.add(index)

print(len(not_ok), not_ok)
            



#print(len(laser_data))
#print(len(robert_data))
#print(len(bert_data))



