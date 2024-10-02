with open('../../data/families_20_dataset/one-hot-families.txt', 'r') as f:
    ls_0 = [i for i in f.read().split('>') if i]
pool = set()
ls = []
for i in ls_0:
    st = i.split('\n')[1]
    if st not in pool:
        pool.add(st)
        ls.append(i)

with open('../../data/families_20_dataset/set-one-hot-families.txt', 'w') as f:
    f.write('>'.join(ls))
