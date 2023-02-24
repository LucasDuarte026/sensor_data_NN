import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

EPOCHS = 1000
input_size = 1
output_size = 1
hidden_layers = 3
hidden_size = 32
learning_rate = 0.01
batch_size = 32
num_epochs = 100
amount = 1000
SAVE = False

# Carregamento dos dados para dentro da rede
voltage_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-21180.dat', sep="\s+")
velo_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-vel-21180.dat', sep="\s+")
data = voltage_fake.assign(
    velocity=velo_fake['velocity']).assign(time=velo_fake['time'])
timer = data['time']
data = data[['voltage', 'velocity']]
data = data.head(amount)
timer = timer.head(amount)

print(data.head(amount))

# Define the MLP architecture as a PyTorch module


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_hidden_layers=2):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x


class VoltageVelocityDataset(Dataset):
    def __init__(self, data):
        self.X = torch.tensor(data['voltage'].values).float().unsqueeze(1)
        self.Y = torch.tensor(data['velocity'].values).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
# Split the data into training and validation sets
train_data = data.sample(frac=0.8, random_state=42)
val_data = data.drop(train_data.index)

# Create datasets and dataloaders for training and validation
train_dataset = VoltageVelocityDataset(train_data)
val_dataset = VoltageVelocityDataset(val_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define the MLP and loss function
mlp = MLP(input_dim=1, output_dim=1, hidden_dim=64, num_hidden_layers=2)
criterion = nn.MSELoss()

# Define the optimizer and learning rate
optimizer = optim.Adam(mlp.parameters(), lr=0.01)

# Train the MLP on the entire dataset
for epoch in range(EPOCHS):
    train_loss = 0.0
    for X, Y in train_loader:
        # Forward pass
        outputs = mlp(X)
        loss = criterion(outputs, Y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.shape[0]

    # Compute validation loss
    with torch.no_grad():
        val_loss = 0.0
        for X, Y in val_loader:
            outputs = mlp(X)
            loss = criterion(outputs, Y)
            val_loss += loss.item() * X.shape[0]

    # Print progress
    if epoch % 100 == 0:
        print("Epoch {}, train loss {:.4f}, val loss {:.4f}".format(epoch, train_loss / len(train_dataset), val_loss / len(val_dataset)))


# Evaluate the MLP on the entire dataset
with torch.no_grad():
    X = torch.tensor(data['voltage'].values).float().unsqueeze(1)
    Y = torch.tensor(data['velocity'].values).float().unsqueeze(1)
    predictions = mlp(X)
    accuracy = ((predictions - Y) ** 2).mean().sqrt().item()
    print("Test accuracy| Mean Loss: {:.4f}".format(accuracy))


# Showing data
shown = pd.DataFrame(predictions.detach().numpy())
shown = shown.assign(original=data['velocity'])
pd.set_option('display.max_rows', None)
print(shown)

# Plotting both the curves simultaneously
plt.plot(timer, data.velocity, color='r', label='data')
plt.plot(timer, predictions.detach().numpy(),
        color='g', label='processed')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Voltage")
plt.ylabel("Velocity")
plt.title("Gráfico com a relação entre Tensão e Velocidade")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()
