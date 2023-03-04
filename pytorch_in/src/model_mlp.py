import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

EPOCHS = 2000
input_size = 1
output_size = 1
hidden_layers = 4
hidden_size = 64
learning_rate = 0.01
batch_size = 32
amount = 1000
SAVE = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print("\tDevice de processamento:",device)

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
print('\n')

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
        self.X = (torch.tensor(
            data['voltage'].values).float().unsqueeze(1)).to(device)
        self.Y = torch.tensor(
            data['velocity'].values).float().unsqueeze(1).to(device)

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
mlp = mlp.to(device)

criterion = nn.MSELoss()

# Define the optimizer and learning rate
optimizer = optim.Adam(mlp.parameters(), lr=0.01)

# Train the MLP on the entire dataset
see_val_loss = np.empty([1, 2])
see_train_loss = np.empty([1, 2])
idx_train = 0
idx_val = 0

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

        see_train_loss = np.append(see_train_loss, [idx_train, loss.item()]).reshape(-1,2)
        train_loss += loss.item() * X.shape[0]
        idx_train = idx_train+1

    # Compute validation loss
    with torch.no_grad():
        val_loss = 0.0
        for X, Y in val_loader:
            outputs = mlp(X)
            loss = criterion(outputs, Y)
            see_val_loss = np.append(see_val_loss, [idx_val, loss.item()]).reshape(-1,2)
            val_loss += loss.item() * X.shape[0]
            idx_val = idx_val+1

    # Print progress
    if epoch % 100 == 0:
        print("| Epoch {:4} | train loss {:4.4f} | val loss {:4.4f} | ".format(
            epoch, train_loss / len(train_dataset), val_loss / len(val_dataset)))


# Evaluate the MLP on the entire dataset
with torch.no_grad():
    X = torch.tensor(data['voltage'].values).float().unsqueeze(1).to(device)
    Y = torch.tensor(data['velocity'].values).float().unsqueeze(1).to(device)
    predictions = mlp(X)
    accuracy = ((predictions - Y) ** 2).mean().sqrt().item()
    print("Test accuracy| Mean Loss: {:.4f}".format(accuracy))


# Showing data
shown = pd.DataFrame(predictions.detach().numpy())
shown = shown.assign(original=data['velocity'])
pd.set_option('display.max_rows', None)
print(shown)


plt.figure(0)
# Plotting both the curves simultaneously
plt.plot(timer, data.velocity, color='r', label='data')
plt.plot(timer, predictions.detach().numpy(),  color='g', label='processed')
plt.xlabel("Voltage")
plt.ylabel("Velocity")
plt.title("Gráfico com a relação entre Tensão e Velocidade")
plt.legend()


print(see_train_loss[1])
figure, axis = plt.subplots(2)

axis[0].plot(see_train_loss[:, 0], see_train_loss[:, 1], color='g', label='train')
plt.legend() 
axis[1].plot(see_val_loss[:, 0], see_val_loss[:, 1], color='r', label='validation')
plt.legend() 

# print(see_train_loss[1])
# plt.figure(1)
# plt.plot(see_train_loss[:, 0], see_train_loss[:, 1], color='g', label='train')
# plt.xlabel("quantity")
# plt.ylabel("loss train")
# plt.title("Loss train em função das interações")
# plt.legend()

# plt.figure(2)
# plt.plot(see_val_loss[:, 0], see_val_loss[:, 1], color='r', label='validation')
# plt.xlabel("quantity")
# plt.ylabel("loss val")
# plt.title("Loss validation  em função das interações")
# plt.legend()

# To load the display window
plt.show()
