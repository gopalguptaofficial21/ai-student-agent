from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

index = faiss.read_index('vector.index')

with open('data/college_info.txt', 'r') as f:
    documents = f.readlines()

documents = [d.strip() for d in documents if d.strip()]

print('AI Student Assistant Ready!')
print('Type your question below. Type exit to quit.')

while True:
    query = input('\nAsk your question: ')
    if query.lower() == 'exit':
        break
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    result = documents[I[0][0]]
    print('\nAnswer:', result)
