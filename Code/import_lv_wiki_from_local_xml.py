import xml.etree.ElementTree as ET
import requests
import time
from constants import path, path_res
from utils import get_wiki_data

# declaration
xml_file_path = path + 'lvwiki-20240401-pages-articles-multistream.xml'
max_pages = 10000
# wikipedia default namespace
namespace = {'ns': 'http://www.mediawiki.org/xml/export-0.10/'}
# variables
i = 0
results = []

if __name__ == '__main__':

    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    start = time.time()
    for page in root.findall('ns:page', namespace):
        if i >= max_pages:
            break

        # extract page title
        id    = page.find('ns:id'      , namespace).text
        title = page.find('ns:title'   , namespace).text
        text  = page.find('ns:revision', namespace).find('ns:text', namespace).text
        # get link to english variant of a page
        data = get_wiki_data(title)
        if data == None:
            continue
        
        f_name = str(i).zfill(6) + '.txt'
        #id;file_name;lv_title;en_title;lv_link;en_link
        results.append(f"{i};{f_name};{title};{data[2]};{data[1]};{data[0]}")
        with open(path_res + 'lv_wikidump\\' + f_name, 'w', encoding="utf-8") as f: 
            f.write(text)

        # debugging
        i += 1
        if i % 100 == 0:
            print(i)

    end = time.time()
    print('time elapsed in minutes', (end - start) / 60)

    with open(path_res + 'results_wikidump.csv', 'w+', encoding="utf-8") as csvfile:
        csvfile.write('id;file_name;lv_title;en_title;lv_link;en_link\n')
        csvfile.write('\n'.join(results))