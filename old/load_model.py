from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer('stsb-xlm-r-multilingual')

with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)
    f.close()
