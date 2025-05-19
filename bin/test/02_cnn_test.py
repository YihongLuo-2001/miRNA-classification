import torch
import torch.nn as nn
# import numpy as np
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
    with torch.no_grad():
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        n_correct, n_samples = 0, 0
        all_predictions, all_labels = [], []

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).to(device)
            _, prediction = torch.max(outputs, 1)

            n_samples += labels.shape[0]
            n_correct += (prediction == labels).sum().item()

            all_predictions.extend(prediction.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy: {acc}%')

        # Calculate the confusion matrix.
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        print(f'Confusion Matrix:\n{conf_matrix}')

        # Calculate precision, recall, and F1 score.
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')


# train_dataset = RNA('../../data/test/miR_T.txt', transform=ToTensor())
test_dataset = RNA('../../data/test/miR_T.txt', transform=ToTensor())
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.manual_seed(0))
model = torch.load('../../models_results/animal_plant/model_cnn.pkl').to(device)

# print('train = ', end='')
# test(model, train_dataset)
print('test = ', end='')
test(model, test_dataset)
