import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Load the data# carregar o banco de dados
voltage_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-21180.dat', sep="\s+")
velo_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-vel-21180.dat', sep="\s+")
data = voltage_fake.assign(
    velocity=velo_fake['velocity']).assign(time=velo_fake['time'])

print(data)
# Define the MLP class with non-linear activation functions

# Split the data into training and test sets
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Extract the input and output variables
X_train = train_data[['voltage']]
X_test = test_data[['voltage']]
y_train = train_data['velocity'].values.reshape(-1, 1)
y_test = test_data['velocity'].values.reshape(-1, 1)

# Define the neural network architecture


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class DeepNN(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers=10, hidden_layer_size=128):
        super(DeepNN, self).__init__()
        self.deep_nn = nn.Sequential()
        for i in range(num_hidden_layers):
            self.deep_nn.add_module(f'ff{i}', nn.Linear(input_size, hidden_layer_size))
            self.deep_nn.add_module(f'activation{i}', nn.ReLU())
            input_size = hidden_layer_size
        self.deep_nn.add_module(f'classifier', nn.Linear(hidden_layer_size, output_size))

    def forward(self, inputs):
        tensor = self.deep_nn(inputs)
        return tensor

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n.squeeze())
        return out
# Instantiate the network and the optimizer
# net = Net()
# net = DeepNN(1,1,3,16)
net = LSTMPredictor(1,10,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Train the network
for epoch in range(500):
    optimizer.zero_grad()
    outputs = net(torch.Tensor(X_train.values))
    loss = criterion(outputs, torch.Tensor(y_train))
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print('Epoch {}, Loss: {:.4f}'.format(epoch, loss.item()))

# Evaluate the network on the test set
y_test_pred = net(torch.Tensor(X_test.values)).detach().numpy()

# Plot the results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_test_pred, color='red')
plt.title('Entrance vs. Output')
plt.xlabel('Entrance')
plt.ylabel('Output')
plt.show()
