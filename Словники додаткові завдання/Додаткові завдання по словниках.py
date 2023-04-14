#K: Номер появи слова
with open('input.txt', 'r') as text:
    d = {}
    result_list = []
    for line in text:
        line = line.strip().lower().replace(';','').replace('.', '').replace(',', '')
        words = line.split()
        for word in words:
            if word not in d:
                result_list.append(0)
                d[word] = 1
            else:
                result_list.append(d[word])
                d[word] += 1
    print(result_list)


#L синоніми
with open('L.txt', 'r') as text:
    for line in text:
        words = line.strip().split()
        it = iter(words)
        res_dct = dict(zip(it, it))
        if 'Goodbye' in res_dct:
            print(res_dct['Goodbye'])
        elif 'Goodbye' in res_dct.values():
            result = [k for k, v in res_dct.items() if v == 'Goodbye'] [0]
            print(result)
        else:
            pass



#N
with open('N.txt', 'r') as f:
    d = {}
    words = f.readlines()[0].strip().split()
    print(words)
    for word in words:
        if word not in d:
            d[word] = 0
        d[word] += 1
print(d)
max_word = ''
max_count = -1
for word, count in sorted(d.items()):
    if count > max_count:
        max_word = word
        max_count = count
print(max_word)



