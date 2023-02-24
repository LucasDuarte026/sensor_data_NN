import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Define the LSTM architecture
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# Define the hyperparameters
input_dim = 1
hidden_dim = 32
num_layers = 1
output_dim = 1
num_epochs = 100
learning_rate = 0.01

# Create the model
model = LSTMPredictor(input_dim, hidden_dim, num_layers, output_dim)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    inputs = torch.autograd.Variable(torch.from_numpy(X_train).float())
    labels = torch.autograd.Variable(torch.from_numpy(y_train).float())

    # Clear the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)

    # Compute the loss
    loss = criterion(outputs, labels)

    # Backward pass
    loss.backward()

    # Update the parameters
    optimizer.step()

    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# Evaluate the model on the test set
with torch.no_grad():
    test_inputs = torch.autograd.Variable(torch.from_numpy(X_test).float())
    test_labels = torch.autograd.Variable(torch.from_numpy(y_test).float())
    test_outputs = model(test_inputs)
    test_loss = criterion(test_outputs, test_labels)
    print('Test Loss: {:.4f}'.format(test_loss.item()))

    # Convert the predictions back to the original scale
    y_pred = test_outputs.data.numpy()
    y_pred = scaler.inverse_transform(y_pred)

    # Plot the results
    plt.plot(X_test, y_test, 'bo', label='True Data')
    plt.plot(X_test, y_pred, 'r', label='Predictions')
    plt.legend()
    plt.show()
