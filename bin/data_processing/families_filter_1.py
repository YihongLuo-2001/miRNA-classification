with open('../../data/families_20_dataset/set-families_dic.csv') as f:
    dic = {i[0]: i[1:] for i in [k.split(',') for k in f.read().split('\n')[1:] if k] if int(i[-2]) < 20}

with open('../../data/families_20_dataset/set-one-hot-families.txt', 'r') as f:
    ls = [(i.split('\t')[1], i) for i in f.read().split('>')]
    ls = [i[1].split('\t')[:1] + dic[i[0]][:-1] + [i[1].split('\t')[2][1:]] for i in ls if i[0] in dic]
    ls = ['\t'.join(i[:3]) + i[-1] for i in ls]
with open('../../data/families_20_dataset/set-one-hot-families-filtered.txt', 'w') as f:
    f.write('>'.join(ls))
