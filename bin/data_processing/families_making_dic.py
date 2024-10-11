def cut(s):
    # print(s)
    flag = 0
    ss = ''
    for i in s:
        if i.isdigit():
            flag = 1
        if flag and i.isdigit():
            ss += i
        elif flag:
            break
    return str(int(ss))


with open('../../data/families_20_dataset/set-one-hot-families.txt') as f:
    ls = [i.split('\n')[0].split('\t')[1] for i in f.read().split('>') if i]

# print(len(ls))
ls_miR = [i for i in ls if ('miR' in i) and ('iab' not in i)]

ls_miR_dic = [(i, 'miR-' + cut(i.split('miR-')[-1].split('miR')[-1].split('-')[0])) for i in ls_miR]
# ls_miR_n = [i for i in ls if 'miR' not in i]
ls_let = [(i, 'let-7') for i in ls if 'let' in i]
ls_other = [(i,
             'lin-4' if 'lin-4' in i else 'bantam' if 'bantam' in i else 'iab-4' if 'iab-4' in i else 'iab-8' if 'iab-8' in i else 'lsy-6' if 'lsy-6' in i else i)
            for i in ls if 'let' not in i and (('miR' not in i) or ('miR' in i and 'iab' in i))]
ls = sorted(ls_miR_dic, key=lambda x: int(x[1].split('-')[1])) + sorted(ls_let, key=lambda x: x[-1]) + sorted(ls_other,
                                                                                                              key=lambda
                                                                                                                  x: x[
                                                                                                                  -1])
dic_count = {i[1]: 0 for i in ls}
for i in ls:
    dic_count[i[1]] += 1
# print(dic_count)
with open('../../data/families_20_dataset/set-families_count.csv', 'w') as f:
    ls_count = sorted([[i, dic_count[i]] for i in dic_count], reverse=True, key=lambda x: int(x[-1]))
    ls_count = [[i[0], str(i[1]), str(n)] for n, i in enumerate(ls_count)]
    title = ['class,count,label']
    f.write('\n'.join(title + [','.join(i) for i in ls_count]))
dic_label = {i[0]: [str(i[2]), str(i[1])] for i in ls_count}
# print(dic_label)
with open('../../data/families_20_dataset/set-families_dic.csv', 'w') as f:
    title = ['name,class,label,count']
    f.write('\n'.join(title + [','.join(list(i) + dic_label[i[-1]]) for i in ls]))
