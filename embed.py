from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

with open('data/college_info.txt', 'r') as f:
    documents = f.readlines()

documents = [d.strip() for d in documents if d.strip()]

embeddings = model.encode(documents)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

faiss.write_index(index, 'vector.index')

print('Embeddings stored successfully!')
