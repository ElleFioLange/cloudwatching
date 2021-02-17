with open('quran-simple.txt', 'r') as f:
    data = f.readlines()

lines = []
prev = None
buffer = ''
for line in data:
    line = line.replace('\n', '')
    if prev == '' and line != '':
        lines.append(buffer)
        buffer = ''
    buffer += line
    prev = line
        

print(lines[:10])