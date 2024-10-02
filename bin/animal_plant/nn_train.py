import torch
import torch.nn as nn
# import numpy as np
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Dataset

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define hyperparameters.
nn_num = 50
input_size = 34 * 4
hidden_size = [2000, 2000, 2000]
output_size = 2

learning_rate = 0.0001
num_epochs = 64
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
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).to(device)
            # prob = [sorted(list(enumerate(i)), key=lambda x:x[1])[::-1][:2] for i in outputs]
            # print([(prob[0][0][0], prob[0][1][0]), (prob[1][0][0], prob[1][1][0])])
            _, prediction = torch.max(outputs, 1)
            # print(prediction, labels)
            n_samples += labels.shape[0]
            n_correct += (prediction == labels).sum().item()
        acc = 100.0 * n_correct / n_samples
        print(f'{acc}%')


train_dataset = RNA('../../data/animal_plant_dataset/rna_train.txt', transform=ToTensor())
test_dataset = RNA('../../data/animal_plant_dataset/rna_test.txt', transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.manual_seed(0))
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # , weight_decay=1e-3)
for epoch in range(num_epochs):
    n_samples, n_correct = 0, 0
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs).to(device)
        _, prediction = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (prediction == labels).sum().item()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('epoch {} / {}, step {} / {}, loss = {:.4f}, accuracy = {:.4f}%'.format(epoch + 1, num_epochs,
                                                                                          i + 1,
                                                                                          len(train_loader),
                                                                                          loss.item(),
                                                                                          100 * n_correct / n_samples))
    print('train = ', end='')
    test(model, train_dataset)
    print('test = ', end='')
    test(model, test_dataset)
torch.save(model, '../../models_results/animal_plant/model_nn.pkl')
# epoch 64 / 64, step 1 / 5, loss = 0.0047, accuracy = 100.0000%
# train = 100.0%
# test = 90.1923076923077%
