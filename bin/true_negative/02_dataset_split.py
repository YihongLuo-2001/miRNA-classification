import random

random.seed(0)
ls = [i for i in open('../../data/true_negative/True-Negative-one-hot.txt', 'r').read().split('>') if i]
random.shuffle(ls)
ls_0, ls_1 = [], []
for n, item in enumerate(ls):
    if n < round(len(ls) * 0.8):
        ls_0.append(item)
    else:
        ls_1.append(item)
f = open('../../data/true_negative/True-Negative-train.txt', 'w')

f.write('>' + '>'.join(ls_0))
f.close()
f = open('../../data/true_negative/True-Negative-test.txt', 'w')
f.write('>' + '>'.join(ls_1))
f.close()
