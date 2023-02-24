import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

LEARNING_RATE = 0.01
EPOCH = 1000

# carregar o banco de dados
voltage_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-21180.dat', sep="\s+")
velo_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-vel-21180.dat', sep="\s+")
data = voltage_fake.assign(
    velocity=velo_fake['velocity']).assign(time=velo_fake['time'])

print(data)
# Define the MLP class with non-linear activation functions


class SISOMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SISOMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()  # Non-linear activation function

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x


# Example usage
# Define the model with 1 input, 10 hidden units, and 1 output
model = SISOMLP(input_size=1, hidden_size=10, output_size=1)
criterion = nn.MSELoss()  # Define the loss function (mean squared error)
# Define the optimizer (Adam with learning rate of 0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Generate some example data (non-linear relationship)
# x = np.linspace(-5, 5, 100)[:, np.newaxis]
# y = np.sin(x) + np.random.normal(0, 0.001, size=x.shape)
x = np.array(data.voltage)    # arrumar os limites da variavel
x = np.reshape(x, (len(x), 1))
y = np.array(data.velocity)   # arrumar os limites da variavel
y = np.reshape(y, (len(y), 1))
# Convert the data to PyTorch tensors
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# Train the model
for epoch in range(EPOCH):
    optimizer.zero_grad()
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, EPOCH, loss.item()))

# Make predictions with the trained model
with torch.no_grad():
    y_pred = model(x_tensor).numpy()

# Plot the results

plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()
