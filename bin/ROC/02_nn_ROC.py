import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define hyperparameters.
nn_num = 50
input_size = 34 * 4
hidden_size = [2000, 2000, 2000]
output_size = 2

learning_rate = 0.0001
num_epochs = 16
batch_size = 512


class ToTensor:
    def __call__(self, sample):
        x, y = sample
        x = torch.tensor([[float(k) for k in i] for i in x])
        return x, torch.tensor(y)


class RNA(Dataset):
    def __init__(self, name, transform=None):
        f = open(name, 'r')
        ls = [[k for k in i.split('\n') if k] for i in f.read().split('>') if i]
        f.close()
        self.x = [i[2:] for i in ls]
        self.y = [int(i[0].split('\t')[-1]) for i in ls]
        self.n_samples = len(self.x)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


# Define NN model.
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.l1 = nn.Linear(input_size, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.l_end = nn.Linear(hidden_size[-1], output_size)

    def forward(self, x):
        x = x.reshape(-1, input_size)
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l_end(out)
        return out


def test(model, test_dataset):
    global batch_size, device, path
    y_score, y_true = [], []
    with torch.no_grad():
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        n_correct, n_samples = 0, 0
        all_predictions, all_labels = [], []

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).to(device)
            y_score += list(F.softmax(outputs, dim=1)[:, 1].to('cpu'))
            y_true += list(labels.to('cpu'))
        return y_true, y_score




train_dataset = RNA('../../data/animal_plant_dataset/rna_train.txt', transform=ToTensor())
test_dataset = RNA('../../data/animal_plant_dataset/rna_test.txt', transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.manual_seed(0))
model = torch.load('../../models_results/animal_plant/model_nn.pkl', weights_only=False).to(device)
output_file_path = '../../models_results/ROC/nn_roc.png'

y_true, y_score = test(model, test_dataset)

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
plt.title('NN model Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig(output_file_path, dpi=600)
plt.close()
