import numpy as np
import faiss
import matplotlib.pyplot as plt
import csv

csv_rows = []
csv_header = ['id', 'distance', 'vector']
csv_name = 'neighbors.csv'

vector_dimensions = 2
vector_amount = 1000
# make reproducable
np.random.seed(1234)
db_vectors = np.random.random((vector_amount, vector_dimensions)).astype('float32')

index = faiss.IndexFlatL2(vector_dimensions)   
index.add(db_vectors)            

# make reproducable
np.random.seed(5678)
query_vectors = np.random.random((1, vector_dimensions)).astype('float32')
k = 4
distances, indices = index.search(query_vectors, k)

plt.scatter(db_vectors[:, 0], db_vectors[:, 1], color='blue')
plt.scatter(query_vectors[:, 0], query_vectors[:, 1], color='red', edgecolors='k')

for i in range(len(distances)):
    for j in range(k):
        neighbor_vector = db_vectors[indices[i][j]]
        csv_rows.append([indices[i][j], distances[i][j], f"{neighbor_vector[0]}, {neighbor_vector[1]}"])
        plt.scatter(neighbor_vector[0], neighbor_vector[1], color='green')

with open(csv_name, 'w+', newline='') as csvfile:
    # creating a csv dict writer object
    writer = csv.writer(csvfile)
    writer.writerow(csv_header)
    writer.writerows(csv_rows)
    csvfile.flush()
    csvfile.close()

plt.show()