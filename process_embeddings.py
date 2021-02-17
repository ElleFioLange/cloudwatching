import numpy as np
import pickle

with open('embeddings.pickle', 'rb') as f:
    data = pickle.load(f)

A = np.array(data[0])
B = np.array(data[1])
C = np.array(data[2])




