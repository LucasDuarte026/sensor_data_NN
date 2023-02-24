import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# read the data
amount = 1000


voltage_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-21180.dat', sep="\s+")
velo_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-vel-21180.dat', sep="\s+")
data = voltage_fake.assign(
    velocity=velo_fake['velocity']).assign(time=velo_fake['time'])
data = data[['voltage', 'velocity']]
data = data.head(amount)

print(data.head(amount))
# normalize the voltage and velocity data using MinMaxScaler
scaler = MinMaxScaler()
data[['voltage', 'velocity']] = scaler.fit_transform(
    data[['voltage', 'velocity']])

# define the custom dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        voltage = self.data.iloc[idx]['voltage']
        velocity = self.data.iloc[idx]['velocity']
        return (voltage, velocity)


# create the dataset and data loader
dataset = CustomDataset(data)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# define the neural network


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x.to(self.fc1.weight.dtype)))
        x = self.fc2(x)
        return x


# initialize the neural network and optimizer
net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# train the neural network
num_epochs = 100
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        data = data.unsqueeze(1)
        target = target.float().unsqueeze(1)
        output = net(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
    print("Epoch:", epoch, "Loss:", loss.item())

# save the trained model
torch.save(net.state_dict(), "./model.pt")
with torch.no_grad():
    result = net(data)
    print(pd.DataFrame(result))
