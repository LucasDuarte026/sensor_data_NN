import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

ACCURACY_TEST = True
RUN = True
LEARNING_RATE = 0.01
EPOCHS = 500

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
####################################################################

# carregar o banco de dados
voltage_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-21180.dat', sep="\s+")
velo_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-vel-21180.dat', sep="\s+")
data = voltage_fake.assign(
    velocity=velo_fake['velocity']).assign(time=velo_fake['time'])

print(data)
####################################################################
# formatar os dados para o formato desejado em pytorch

X = np.array(data.voltage)    # arrumar os limites da variavel
X = np.reshape(X, (len(X), 1))
Y = np.array(data.velocity)   # arrumar os limites da variavel
Y = np.reshape(Y, (len(Y), 1))

n_samples, n_features = X.shape
n_output_samples, output_size = Y.shape

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0, shuffle=False)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

Y_train = Y_train.view(Y_train.shape[0], 1)
Y_test = Y_test.view(Y_test.shape[0], 1)
print(f'X_train{X_train}\nY_train{Y_train}')

####################################################################

# Classes de treino


class SimpleModel(nn.Module):
    def __init__(self, n_input_features):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, X):
        y_pred = torch.sigmoid(self.linear(X))
        return y_pred


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


model = DeepNN(n_features, output_size, 3, 8)
# model = SimpleModel(n_features)
num_epochs = EPOCHS
learning_rate = LEARNING_RATE
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
if RUN == True:
    for epoch in range(num_epochs):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, Y_train)

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        if (epoch+1) % 10 == 0:
            pass
            print(f'\repoch: {epoch+1}, loss = {loss.item():.8f}')

    if ACCURACY_TEST == True:
        with torch.no_grad():
            y_predicted = model(X_test)
            y_predicted_cls = y_predicted.round()
            # y_predicted_cls = y_predicted

            print(
                f'y_predicted\n{y_predicted}\ny_predicted_cls\n{y_predicted_cls}')

            acc = y_predicted_cls.diff(Y_test).sum()/float(Y_test.shape[0])
            print(f'accuracy: {acc.item():.4f}')


result = pd.DataFrame((model.forward(X_train))).assign(
    velocity=velo_fake['velocity'])
print(f'\nresult: \n{result}')
# plt.plot(data.time,data.voltage)

plt.plot(data.time[:96000], result)
plt.plot(data.time, data.velocity)
# plt.show()
