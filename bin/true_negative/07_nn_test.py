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


train_dataset = RNA('../../data/true_negative/True-Negative-train.txt', transform=ToTensor())
test_dataset = RNA('../../data/true_negative/True-Negative-test.txt', transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.manual_seed(0))
model = torch.load('../../models_results/true_negative/model_nn.pkl', weights_only=False).to(device)

print('train = ', end='')
test(model, train_dataset)
print('test = ', end='')
test(model, test_dataset)
