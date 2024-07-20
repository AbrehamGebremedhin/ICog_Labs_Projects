import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Initialize MongoDB client
client = MongoClient("mongodb://localhost:27017/")
db = client["embedding_models"]
collection = db["comparisons"]

db.comparisons.create_search_index(
    "indexed_embedding",
    "vectorSearch",
    {
        "fields": {
            "type": "vector",
            "path": "embeddings",
            "dimension": 512,
            "similarity": "cosine"
        }
    }
)


# Retrieve embeddings from MongoDB
def retrieve_embeddings(model_id):
    document = collection.find_one({"model_id": model_id})
    embeddings = np.array(document["embeddings"])
    process_time = document["process_time"]
    return embeddings, process_time


# Compare Cosine Similarity
def compare_cosine_similarity(embeddings_dict):
    for model_id, embeddings in embeddings_dict.items():
        sim_matrix = cosine_similarity(embeddings)
        avg_sim = np.mean(sim_matrix)
        print(f"Average cosine similarity for {model_id}: {avg_sim:.4f}")


# Visualize Embeddings with t-SNE
def visualize_embeddings(embeddings_dict):
    for model_id, embeddings in embeddings_dict.items():
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        plt.figure(figsize=(8, 6))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
        plt.title(f"t-SNE Visualization of {model_id}")
        plt.show()


def visualize_embeddings_3d(embeddings_dict):
    for model_id, embeddings in embeddings_dict.items():
        tsne = TSNE(n_components=3, random_state=42)
        embeddings_3d = tsne.fit_transform(embeddings)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], alpha=0.5)
        ax.set_title(f"3D t-SNE Visualization of {model_id}")
        plt.show()


model_ids = [
    "microsoft/codebert-base",
    "intfloat/multilingual-e5-large",
    "jinaai/jina-embeddings-v2-base-en",
    "mixedbread-ai/mxbai-embed-large-v1",
    "facebook/bart-large"
]

embeddings_dict = {}
process_times = {}

for model_id in model_ids:
    embeddings, process_time = retrieve_embeddings(model_id)
    embeddings_dict[model_id] = embeddings
    process_times[model_id] = process_time

print("Processing times:")
for model_id, process_time in process_times.items():
    print(f"{model_id}: {process_time:.2f} seconds")

compare_cosine_similarity(embeddings_dict)
visualize_embeddings(embeddings_dict)
visualize_embeddings_3d(embeddings_dict)
