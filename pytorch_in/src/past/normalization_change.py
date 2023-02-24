import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Carregamento dos dados para dentro da rede
amount =100
voltage_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-21180.dat', sep="\s+")
velo_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-vel-21180.dat', sep="\s+")
data = voltage_fake.assign(
    velocity=velo_fake['velocity']).assign(time=velo_fake['time'])
data = data[['voltage', 'velocity']]
data = data.head(amount)

print(data.head(amount))

# Split the data into train, test, and validation sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Normalize the data using PyTorch's Normalize
normalize = lambda x: (x - x.mean()) / x.std()
train_data = normalize(train_data)
test_data = normalize(test_data)
val_data = normalize(val_data)

# Convert the data into PyTorch tensors
train_data = torch.Tensor(train_data.values)
test_data = torch.Tensor(test_data.values)
val_data = torch.Tensor(val_data.values)

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_size):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for i in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = nn.ReLU()(layer(x))
        x = self.layers[-1](x)
        return x

# Set the hyperparameters
input_size = 1
output_size = 1
hidden_layers = 2
hidden_size = 32
learning_rate = 0.01
batch_size = 32
num_epochs = 100

# Instantiate the neural network
net = Net(input_size, output_size, hidden_layers, hidden_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Train the neural network
for epoch in range(num_epochs):
    for i in range(0, train_data.size(0), batch_size):
        batch = train_data[i:i+batch_size, :]
        inputs = batch[:, :-1]
        labels = batch[:, -1].view(-1, 1)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print('Epoch: {}, Loss: {:.6f}'.format(epoch+1, loss.item()))

# Evaluate the neural network on the test set
net.eval()
with torch.no_grad():
    inputs = test_data[:, :-1]
    labels = test_data[:, -1].view(-1, 1)
    outputs = net(inputs)
    test_loss = criterion(outputs, labels)
print('Test Loss: {:.6f}'.format(test_loss.item()))

# Denormalize the predicted and true labels
pred_labels = outputs.detach().numpy() * train_data.std(axis=0)[-1] + train_data.mean(axis=0)[-1]
true_labels = labels.numpy() * train_data.std(axis=0)[-1] + train_data.mean(axis=0)[-1]

# Compute the test accuracy
test_accuracy = 1 - abs(true_labels - pred_labels).mean() / true_labels.mean()
print('Test Accuracy: {:.2f}%'.format(test_accuracy*100))

# Save the trained model
torch.save(net.state_dict(), 'model.pth')
