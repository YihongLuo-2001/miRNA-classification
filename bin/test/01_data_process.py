def one_hot(seq):
    # dic = {'A': '1000', 'U': '0100', 'C': '0010', 'G': '0001', 'N': '0000'}
    dic = 'AUCG'
    seq = seq + '\n' + '\n'.join(
        [''.join([str(int(dic[i] == k)) for k in seq.split('\n')[-1]]) for i in range(4)])
    return seq


path = '../../data/test/trans_PARE_p1.tsv'
path_0 = '../../data/test/miRNA.fa_GSTAr.tsv'
path_output = '../../data/test/miR_T.txt'
path_output_0 = '../../data/test/miR_F.txt'
with open(path) as f:
    ls = [i.split('\t') for i in f.read().split('\n') if i]
dic_keys = ls[0]
dic_ls = []
for i in ls[1:]:
    dic_ls.append({dic_keys[num]: item for num, item in enumerate(i)})
seqs = [(i['Query'], i['Sequence'].split('&')[0].replace('-', '')) for i in dic_ls if
        float(i['DegradomePval']) < 0.05 and i['DegradomeCategory'] == '0']
seqs = [i for i in seqs if len(i[-1]) < 35]

with open(path_0) as f:
    ls_0 = [i.split('\t') for i in f.read().split('\n') if i]
dic_keys_0 = ls_0[0]
dic_ls_0 = []
for i in ls_0[1:]:
    dic_ls_0.append({dic_keys_0[num]: item for num, item in enumerate(i)})
seqs_0 = [(i['Query'], i['Sequence'].split('&')[0].replace('-', '')) for i in dic_ls_0]
seqs_0 = [i for i in seqs_0 if len(i[-1]) < 35]
a = set(seqs)
all_seq = set(seqs_0)
b = all_seq - a
t = [one_hot('>{}\tPlant\t1\n{:N<34}'.format(i[0], i[-1])) for i in a]
f = [one_hot('>{}\tPlant\t1\n{:N<34}'.format(i[0], i[-1])) for i in b]
st_0 = '\n'.join(f)
st = '\n'.join(t)
with open(path_output, 'w') as f:
    f.write(st)
with open(path_output_0, 'w') as f:
    f.write(st_0)
