import os
import csv
from constants import path_search, path_search_FLAT

def count_true_in_file(filepath):
    count = 0
    with open(filepath, 'r') as file:
        for line in file:
            count += line.count('True')
    return count

#type = 'HNSW'
type = 'FLAT'

if type == 'HNSW':
    path = path_search
elif type == 'FLAT':
    path = path_search_FLAT
else:
    print('Wrong type...')
    exit()

result_file = f"Results_{type}.csv"
output_file =  path + result_file
i = 0
total = 77736

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(["model", "type", "kNN", "successful_finds", "recall", "recall_percentage", "recall_percentage_rounded"])
    
    for root, dirs, files in os.walk(path):
        # only csv, not output file, then sort by name length, if length same, then alhpabetically.
        files = sorted([f for f in files if f.endswith('.csv') and f != result_file], key=lambda x: (len(x), x))
        for file in files:
            filepath = os.path.join(root, file)
            count = count_true_in_file(filepath)
            
            relative_path = filepath.split(os.path.sep)[-3::]
            model = relative_path[0]
            type  = relative_path[1]
            kNN   = relative_path[2].split('.')[0]

            recall = count / total
            recall_percentage = recall * 100
            recall_percentage_rounded = round(recall_percentage, 2)

            writer.writerow([model, type, kNN, count, recall, recall_percentage, recall_percentage_rounded])