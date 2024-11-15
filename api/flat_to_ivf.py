import numpy as np
import faiss
from tqdm import tqdm
import pdb

# Load the flat index
flat_index = faiss.read_index("/checkpoint/amaia/explore/comem/data/scaling_out/embeddings/facebook/dragon-plus-context-encoder/dpr_wiki/8-shards/index/0_1_2_3_4_5_6_7/index.faiss")

# Get the number of vectors and their dimensionality
num_vectors = flat_index.ntotal
d = flat_index.d

# Extract the embeddings using the reconstruct method
embeddings = np.zeros((num_vectors, d), dtype='float32')
for i in tqdm(range(num_vectors)):
    embeddings[i] = flat_index.reconstruct(i)
    if i > 10:
        break
pdb.set_trace()
    
# # Create and train an IVF index
# nlist = 100  # number of centroids
# quantizer = faiss.IndexFlatIP(d)  # using Inner Product (IP) metric
# ivf_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
# ivf_index.train(embeddings)

# # Add embeddings to the IVF index
# ivf_index.add(embeddings)

# # Save the IVF index to a file
# faiss.write_index(ivf_index, "/checkpoint/amaia/explore/comem/data/scaling_out/embeddings/facebook/dragon-plus-context-encoder/dpr_wiki/8-shards/index/0_1_2_3_4_5_6_7/ivf_flat_index.faiss")


# Load IVF index
ivf_index = faiss.read_index("/checkpoint/amaia/explore/comem/data/scaling_out/embeddings/facebook/dragon-plus-context-encoder/dpr_wiki/8-shards/index/0_1_2_3_4_5_6_7/ivf_flat_index.faiss")
# Perform a search
ivf_index.nprobe = 100
# query_vector = np.random.random((1, d)).astype('float32')
query_vector = embeddings[0].reshape(1, d) 
distances, indices = ivf_index.search(query_vector, k=5)

print("Distances:", distances)
print("Indices:", indices)

