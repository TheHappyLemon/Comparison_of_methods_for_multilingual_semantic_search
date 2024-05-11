import numpy as np
import faiss
import matplotlib.pyplot as plt

res = []

dimensions = 2
vector_amount = 1000
# make reproducable
np.random.seed(1234)
db_vectors = np.random.random((vector_amount, dimensions)).astype('float32')

index = faiss.IndexHNSWFlat(dimensions, 128)   
index.add(db_vectors)            

np.random.seed(5678)
query_vectors = np.random.random((1, dimensions)).astype('float32')
print(query_vectors)
distances, indices = index.search(query_vectors, 5)

print(distances)
print(indices)

plt.scatter(db_vectors[:, 0], db_vectors[:, 1], color='blue')
plt.scatter(query_vectors[:, 0], query_vectors[:, 1], color='red', edgecolors='k')

for i in range(len(distances)):
    for j in range(len(db_vectors[indices[i]])):
        neighbor_vector = db_vectors[indices[i][j]]
        res.append([indices[i][j], distances[i][j], f"{neighbor_vector[0]}, {neighbor_vector[1]}"])
        plt.scatter(neighbor_vector[0], neighbor_vector[1], color='green')

print(res)
plt.show()