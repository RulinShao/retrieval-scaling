import torch
import faiss
import numpy as np
import argparse
import pdb
import contriever.src.contriever
from src.search import embed_queries
from src.embed import embed_passages




device = 'cuda' if torch.cuda.is_available()  else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', default='facebook/contriever-msmarco')
parser.add_argument('--no_title', default=False)
parser.add_argument('--lowercase', default=False)
parser.add_argument('--normalize_text', default=False)
parser.add_argument('--per_gpu_batch_size', default=1024)
parser.add_argument('--passage_maxlength', default=256)
parser.add_argument('--question_maxlength', default=256)
args = parser.parse_args()


query_encoder, query_tokenizer, _ = contriever.src.contriever.load_retriever('facebook/contriever-msmarco')
query_encoder = query_encoder.to(device)



documents = [
    {"id": 0, "text": "The quick brown fox jumps over the lazy dog."},
    {"id": 1, "text": "Never stop learning because life never stops teaching."},
    {"id": 2, "text": "The purpose of our lives is to be happy."},
]

query = "Life is full of surprises."



passage_ids, passage_embedding =  embed_passages(args, documents, query_encoder, query_tokenizer)  # document encoder is the same as query encoder
query_embedding = embed_queries(args, [query], query_encoder, query_tokenizer, args.model_name_or_path)


pdb.set_trace()

dimension = passage_embedding.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatIP(dimension)  # Flat index with inner product (dot product) metric

# Add passage embeddings to the index
index.add(passage_embedding)
# Search the index with the query embedding
k = 3  # Number of nearest neighbors to retrieve
distances, indices = index.search(query_embedding, k)
# Print the results
print("Nearest neighbors:")
for i in range(k):
    print(f"Document ID: {passage_ids[indices[0][i]]}, Distance: {distances[0][i]}")


# Compute inner products manually
print("\nManual Inner Products:")
manual_inner_products = np.dot(passage_embedding, query_embedding.T).flatten()
for i, inner_product in enumerate(manual_inner_products):
    print(f"Document ID: {passage_ids[i]}, Inner Product: {inner_product}")



# Create an IVF_Flat index
nlist = 1  # Number of centroids (clusters)
ivf_index = faiss.IndexIVFFlat(index, dimension, nlist) #, faiss.METRIC_INNER_PRODUCT)
# Train the IVF index (necessary step before adding vectors)
ivf_index.train(passage_embedding)
# Add passage embeddings to the IVF index
ivf_index.add(passage_embedding)
# Search the IVF index with the query embedding
k = 3  # Number of nearest neighbors to retrieve
ivf_distances, ivf_indices = ivf_index.search(query_embedding, k)
# Print the results from IVF_Flat
print("IVF_Flat Nearest neighbors:")
for i in range(k):
    print(f"Document ID: {passage_ids[ivf_indices[0][i]]}, Distance: {ivf_distances[0][i]}")
# Compute inner products manually
print("\nManual Inner Products:")
manual_inner_products = np.dot(passage_embedding, query_embedding.T).flatten()
for i, inner_product in enumerate(manual_inner_products):
    print(f"Document ID: {passage_ids[i]}, Inner Product: {inner_product}")


print("\nManual Euclidean Distances:")
manual_euclidean_distances = np.linalg.norm(passage_embedding - query_embedding, axis=1)
for i, euclidean_distance in enumerate(manual_euclidean_distances):
    print(f"Document ID: {passage_ids[i]}, Euclidean Distance: {euclidean_distance}")
    
    
pdb.set_trace()
