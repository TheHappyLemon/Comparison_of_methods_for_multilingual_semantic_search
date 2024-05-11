import csv
from constants import path_res, path_search, path_search_FLAT
import h5py 

def write_to_csv(same_titles):
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
    errors = []
    for same_title in same_titles:
        i = i + 1
        en_contents = None
        lv_contents = None
        with open(path_res + 'en_cirrussearch_title\\' + same_title[0], 'r', encoding='utf-8') as en_file:
            en_contents = en_file.read()
        with open(path_res + 'lv_cirrussearch_title\\' + same_title[0], 'r', encoding='utf-8') as en_file:
            lv_contents = en_file.read()
        if lv_contents != en_contents:
            err = f"File '{same_title[0]}' equals to '{en_contents}' in english and '{lv_contents}' in latvian! But should be same..."
            errors.append(err)
    return errors

def get_pages_not_found_perfectly(filenames, file_path):
    was_unable_to_find_perfectly = []
    with open(file_path, 'r', newline='\n', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            if row[0] in filenames:
                if row[1] == 'False' or row[3] != '0.0':
                    was_unable_to_find_perfectly.append(row[0])
    return was_unable_to_find_perfectly

def get_different_embeddings(not_found_perfectly, model):
    with h5py.File(path_res + 'embeddings.hdf5', 'r') as file:
        mapping_set = file['/mapping']
        hdf5_indexes = []
        for i in range(len(mapping_set)):
            data = mapping_set[i][0].decode()
            if data in not_found_perfectly:
                hdf5_indexes.append(i)
        #print(len(hdf5_indexes), len(not_found_perfectly))
        #print('Comparing embeddings...')
        not_ok = set()
        for index in hdf5_indexes:
            lv_embd = file[f'/{model}/title/lv_cirrussearch'][index]
            en_embd = file[f'/{model}/title/en_cirrussearch'][index]
            for i in range(len(lv_embd)):
                difference = abs(lv_embd[i] - en_embd[i])
                if difference != 0.0:
                    #print(lv_embd[i], en_embd[i], difference)
                    not_ok.add(index)
    return not_ok

#path = path_search
path = path_search_FLAT

# Here pages with same title on both languages are extracted and checked if files really contain same data
same_titles = get_pages_with_same_title( path_res + 'results_cirrussearch_ALL_common.csv')
not_same = check_if_contents_are_really_the_same(same_titles)
if len(not_same) > 0:
    print('Errors:')
    for err in not_same:
        print(err)
write_to_csv(same_titles)
filenames = [data[0] for data in same_titles]

# With the same model same text on both languages must have same embeddings, so when searching, distance between them must be 0.0
# so take row where search was performed on page with same texts and if file was either not found or distance is not 0.0, it is some kind of error
# to make sure that it is not embeddings erros, check embeddings itself, each vector position.
laser_data = get_pages_not_found_perfectly(filenames, path + 'LASER\\title\\1NN.csv')
print(f'In "{path}" with LASER model "{len(laser_data)}" pages with same texts were not found perfectly')
diff = get_different_embeddings(laser_data , 'LASER')
print(f'Out of those "{len(laser_data)}" imperfections "{len(diff)}" embeddings differ between en and lv')

robert_data = get_pages_not_found_perfectly(filenames, path + 'XLM-RoBERTa\\title\\1NN.csv')
print(f'In "{path}" with XLM-RoBERTa model "{len(robert_data)}" pages with same texts were not found perfectly')
diff = get_different_embeddings(robert_data, 'XLM-RoBERTa')
print(f'Out of those "{len(robert_data)}" imperfections "{len(diff)}" embeddings differ between en and lv')

bert_data = get_pages_not_found_perfectly(filenames, path + 'BERT\\title\\1NN.csv')
print(f'In "{path}" with BERT model "{len(bert_data)}" pages with same texts were not found perfectly')
diff = get_different_embeddings(bert_data  , 'Bert')
print(f'Out of those "{len(bert_data)}" imperfections "{len(diff)}" embeddings differ between en and lv')
