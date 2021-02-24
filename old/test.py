import pickle
from sentence_transformers import SentenceTransformer

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

love_emb = model.encode('love')
hate_emb = model.encode('hate')
faith_emb = model.encode('faith')


print(love_emb, hate_emb, faith_emb)
print(len(love_emb))