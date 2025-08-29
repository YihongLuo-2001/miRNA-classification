import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV


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


# Path
# train_file_path = '../../data/animal_plant_dataset/rna_train.txt'
test_file_path = '../../data/animal_plant_dataset/rna_test.txt'
model_file_path = '../../models_results/animal_plant/svm_model_6.joblib'

# X_train, y_train = load_data_v3(train_file_path)
X_test, y_test = load_data_v3(test_file_path)
output_file_path = '../../models_results/ROC/svm_roc.png'

model = joblib.load(model_file_path)
y_true = y_test
y_score = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

print("FPR:", fpr)
print("TPR:", tpr)
print("Thresholds:", thresholds)
print(f"AUC = {roc_auc:.4f}")

# Youden index: maximize TPR - FPR
youden_index = tpr - fpr
best_idx = np.argmax(youden_index)
best_threshold = thresholds[best_idx]
print(f"Best threshold (Youden): {best_threshold:.4f}, TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f}")

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, label='Random')
plt.scatter(fpr[best_idx], tpr[best_idx], c='red', label=f'Best thresh={best_threshold:.2f}')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM model Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig(output_file_path, dpi=600)
plt.close()

