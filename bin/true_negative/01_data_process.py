import random


def one_hot(seq):
    # dic = {'A': '1000', 'U': '0100', 'C': '0010', 'G': '0001', 'N': '0000'}
    dic = 'AUCG'
    seq = seq + '\n' + '\n'.join(
        [''.join([str(int(dic[i] == k)) for k in seq.split('\n')[-1]]) for i in range(4)])
    return seq


random.seed(0)
neg_num = 75432
head = '>neg\tFalse\t0\n'
path = '../../data/true_negative/True-one-hot.txt'
path_0 = '../../data/true_negative/True-Negative-one-hot.txt'
with open(path) as f:
    seqs_detail = ['>' + '\n'.join([k for k in i.split('\n') if k]) for i in f.read().split('>') if i]
    seqs = [i.split('\n')[1] for i in seqs_detail]
seqs_random, seqs_random_detail = [], []
for num, item in enumerate(seqs):
    while True:
        seq = ''.join([i if i == 'N' else ('A', 'U', 'C', 'G')[random.randrange(4)] for i in item])
        if seq not in seqs_random and seq not in seqs:
            seqs_random.append(seq)
            seqs_random_detail.append(head + one_hot(seq))
            break
    print('{}/{}\t{}'.format(num + 1, len(seqs), seq))
with open(path_0, 'w') as f:
    f.write('\n'.join(seqs_random_detail + seqs_detail))

