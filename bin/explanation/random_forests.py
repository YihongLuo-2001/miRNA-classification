import joblib
from sklearn.tree import export_graphviz


def _reduce_by_keys(foo, double_tuples):
    dic = {}
    for i in double_tuples:
        dic[i[0]] = i[1] if i[0] not in dic else foo(dic[i[0]], i[1])
    return dic.items()


def track(tree, road, node=0):
    global feature_names
    road = [i for i in road.split('_') if i]
    m_feature = tree.tree_.feature
    if node < 0:
        return 'leaf', '_'.join(road)
    if len(road) == 0:
        return feature_names[m_feature[node]], tree.tree_.value[node][0][1] / (
                tree.tree_.value[node][0][0] + tree.tree_.value[node][0][1])
    left, right = tree.tree_.children_left, tree.tree_.children_right
    node = left[node] if road[0] == 'l' else right[node]
    road = '_'.join(road[1:])
    return track(tree, road, node)


model = joblib.load('../../models_results/animal_plant/best_random_forest_model_5.joblib')
feature_names = [['{}_{}'.format(i // 4 + 1, k) for k in ['A', 'U', 'C', 'G']][i % 4] for i in range(34 * 4)]

# Statistical path information.
roads = ['', 'l', 'r', 'l_l', 'l_r', 'r_l', 'r_r', 'l_l_l', 'l_l_r', 'l_r_l', 'l_r_r', 'r_l_l', 'r_l_r', 'r_r_l',
         'r_r_r']
roads_0 = ['root'] + [i.upper() for i in roads[1:]]
with open('../../models_results/explanation/trees.tsv', 'w') as f:
    title = ['{}\tfeature_importance\tclass_current_depth'.format(i) for i in roads_0]
    f.write('\t'.join(['tree_id', 'leaves_num', 'depth', 'nodes_num'] + title) + '\n')
for n, i in enumerate(model):
    nodes = [track(i, k) for k in roads]
    # nodes = [track(i, 'l_l_l_l_l_l_l_l_l_l_l_l_l_l_l_l_l_l_l_l_l_l_l_l_l_l_l_l_l_l_l')]
    ls = [n + 1, i.get_n_leaves(), i.get_depth(), i.tree_.node_count] + [
        '{}<=0.5\t{}\t{}'.format(k[0], k[1], 'leaf' if not k[1] else 'Animal' if float(k[1]) <= 0.5 else 'Plant') for k
        in
        nodes]
    ls = '\t'.join([str(k) for k in ls])
    with open('../../models_results/explanation/trees.tsv', 'a') as f:
        f.write(ls + '\n')
a = []
with open('../../models_results/explanation/position.tsv', 'w') as f:
    f.write('\t'.join(['criteria', 'total_feature_importance']) + '\n')
for num, item in enumerate(model):
    a += [(feature_names[n], i) for n, i in enumerate(item.tree_.compute_feature_importances())]
dic = _reduce_by_keys(lambda x, y: x + y, a)
for i in sorted(dic, key=lambda x: x[-1])[::-1]:
    with open('../../models_results/explanation/position.tsv', 'a') as f:
        f.write('{}\t{}\n'.format(i[0], i[1]))

# Generate dot files for all decision trees in the random forest.
for num in range(1, 2001):
    print('{}/2000'.format(num), end='')
    # max_depth sets the number of layers for visualizing the decision tree.
    export_graphviz(model[num - 1], out_file='../../models_results/explanation/trees_dot/tree_{}.dot'.format(num), feature_names=feature_names,
                    class_names=['Animal', 'Plant'], filled=True, rounded=True, max_depth=3)
    print('\r', end='')
