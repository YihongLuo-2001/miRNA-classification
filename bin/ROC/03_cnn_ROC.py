import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define hyperparameters.

# input_size = 34 * 4
output_size = 2
hidden_layer = 20000

learning_rate = 0.001
num_epochs = 16
batch_size = 256


class ToTensor:
    def __call__(self, sample):
        x, y = sample
        x = [[float(k) for k in i] for i in x]
        x = torch.tensor([x]).reshape(1, 8, -1)
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


# Define CNN model.
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 1 * 2, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 128 * 1 * 2)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


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
output_file_path = '../../models_results/ROC/cnn_roc.png'
model = torch.load('../../models_results/animal_plant/model_cnn.pkl', weights_only=False).to(device)

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
plt.title('CNN model Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig(output_file_path, dpi=600)
plt.close()
