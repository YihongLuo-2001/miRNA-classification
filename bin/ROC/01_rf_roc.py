import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

np.random.seed(0)


def load_data_v3(file_path):
    """Load the data according to the new file format and separate the features and labels. """
    print(f"Loading data...：{file_path}")
    with open(file_path, 'r') as file:
        content = file.read()

    parts = content.strip().split('>')[1:]
    X, y = [], []

    for part in parts:
        lines = part.strip().split('\n')
        label = int(lines[0].split('\t')[-1])
        y.append(label)

        # Every four lines represent the complete encoding of a nucleotide sequence, processed column by column.
        one_hot_encoded = np.array([list(map(int, line)) for line in lines[2:6]]).T
        X.append(one_hot_encoded)

    print(
        f"Complete loading! Data set: {len(X)} samples, each sample has {len(X[0])} types of nucleotide bases, the vector length for each nucleotide base is {len(X[0][0])}")
    return np.array(X).reshape(len(X), -1), np.array(y)


def compute_roc(predictions, true_labels=None):
    """
    计算ROC曲线和AUC值

    参数:
    predictions -- 列表，包含每个样本的预测概率值(0到1之间)
    true_labels -- 列表，包含每个样本的真实标签(0或1)。如果为None，则生成模拟标签

    返回:
    fpr -- 假正率数组
    tpr -- 真正率数组
    thresholds -- 阈值数组
    roc_auc -- AUC值
    """
    # 如果没有提供真实标签，生成模拟标签（假设前一半为负类，后一半为正类）
    if true_labels is None:
        n = len(predictions)
        true_labels = [0] * (n // 2) + [1] * (n - n // 2)
        print("使用模拟标签: 前一半为0(负类)，后一半为1(正类)")

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, thresholds, roc_auc


def plot_roc_curve(fpr, tpr, roc_auc):
    """绘制ROC曲线"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title('接收者操作特征(ROC)曲线')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


# 示例使用
if __name__ == "__main__":
    # Path
    train_file_path = '../../data/animal_plant_dataset/rna_train.txt'
    test_file_path = '../../data/animal_plant_dataset/rna_test.txt'
    output_file_path = '../../models_results/ROC/random_forest_roc.png'
    model_file_path = '../../models_results/animal_plant/best_random_forest_model_5.joblib'

    model = joblib.load(model_file_path)
    # x, y = load_data_v3(train_file_path)
    x, y = load_data_v3(test_file_path)
    result_forest = []
    for num, model_f in enumerate(model):
        result_forest.append(model_f.predict(x))
        print('\r{}/{}'.format(num, len(model)), end='')
    y_score = [i for i in np.mean(np.array(result_forest), axis=0)]
    y_true = y
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    print("FPR:", fpr)
    print("TPR:", tpr)
    print("Thresholds:", thresholds)
    print(f"AUC = {roc_auc:.4f}")

    # 找最佳阈值（Youden index: maximize TPR - FPR）
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_idx]
    print(f"Best threshold (Youden): {best_threshold:.4f}, TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f}")

    # 画 ROC 曲线
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, label='Random')
    plt.scatter(fpr[best_idx], tpr[best_idx], c='red', label=f'Best thresh={best_threshold:.2f}')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('RF model Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(output_file_path, dpi=600)
    plt.close()

