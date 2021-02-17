import pickle

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)
    f.close()

kj_sent = []
with open('king-james.txt', 'r') as f:
    data = f.readlines()
    lines=[]
    prev=None
    buffer=''
    for line in data:
        line = line.replace('\n', '')
        if prev == '' and line != '':
            lines.append(buffer)
            buffer = ''
        buffer += line
        prev = line
    kj_sent = lines

q_sent = []
with open('quran-simple.txt', 'r') as f:
    data = f.readlines()
    q_sent=data

sentences = kj_sent + q_sent

sentence_embeddings = model.encode(sentences)

with open ('embeddings.pickle', 'wb') as f:
    pickle.dump(sentence_embeddings, f)
    f.close()