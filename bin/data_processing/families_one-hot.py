def one_hot(seq):
    # dic = {'A': '1000', 'U': '0100', 'C': '0010', 'G': '0001', 'N': '0000'}
    dic = 'AUCG'
    seq = seq + '\n' + '\n'.join(
        [''.join([str(int(dic[i] == k)) for k in seq.split('\n')[-1]]) for i in range(4)])
    return seq


ls_0 = [one_hot('>{}\t{}\t0\n{:N<34}'.format(' '.join(i.split('\t')[0].split()[:2]), i.split('\t')[0].split()[-2],
                                             i.split('\t')[0].split()[-1])) for i in
        open('../../data/families_20_dataset/Animal_miRNA.txt', 'r').read().split('\n') if i]
ls_1 = [one_hot('>{}\t{}\t1\n{:N<34}'.format(i.split('\t')[0], i.split('\t')[-3], i.split('\t')[-2])) for i in
        open('../../data/families_20_dataset/Plant_miRNA.txt', 'r').read().split('\n') if i]
# print((max([len(i.split('\n')[1].replace('N', '')) for i in ls_1])))
# print(ls_1[-1])
ls = '\n'.join(ls_0 + ls_1)
f = open('../../data/families_20_dataset/one-hot-families.txt', 'w')
f.write(ls)
f.close()
