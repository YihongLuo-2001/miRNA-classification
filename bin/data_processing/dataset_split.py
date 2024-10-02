import random

random.seed(0)
ls = [i for i in open('../../data/families_20_dataset/set-one-hot-families-filtered.txt', 'r').read().split('>') if i]
random.shuffle(ls)
ls_0, ls_1 = [], []
for n, item in enumerate(ls):
    if n < round(len(ls) * 0.8):
        ls_0.append(item)
    else:
        ls_1.append(item)
f = open('../../data/families_20_dataset/set_train_families-filtered.txt', 'w')

f.write('>' + '>'.join(ls_0))
f.close()
f = open('../../data/families_20_dataset/set_test_families-filtered.txt', 'w')
f.write('>' + '>'.join(ls_1))
f.close()
