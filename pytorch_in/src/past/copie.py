# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import StandardScaler

ACCURACY_TEST = True
LEARNING_RATE = 0.01
EPOCHS = 100


# carregar o banco de dados
voltage_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-21180.dat', sep="\s+")
velo_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-vel-21180.dat', sep="\s+")
data = voltage_fake.assign(velocity=velo_fake['velocity'])
# print(data)
####################################################################
# formatar os dados para o formato desejado em pytorch

X = np.array(data.voltage) / 10  # arrumar os limites da variavel
X = np.reshape(X, (len(X), 1))
Y = np.array(data.velocity)/100  # arrumar os limites da variavel
Y = np.reshape(Y, (len(Y), 1))

print(f'X: {X}\nY: {Y}')

n_samples, n_features = X.shape

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1234)

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# print(f'X_train{X_train}\nX_test{X_test}')

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

Y_train = Y_train.view(Y_train.shape[0], 1)
Y_test = Y_test.view(Y_test.shape[0], 1)
print(f'X_train{X_train}\Y_train{Y_train}')


class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, X):
        y_pred = torch.sigmoid(self.linear(X))
        return y_pred


model = Model(n_features)
num_epochs = EPOCHS
learning_rate = LEARNING_RATE
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, Y_train)

    loss.backward()
    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        pass
        print(f'\repoch: {epoch+1}, loss = {loss.item():.4f}')
        
if ACCURACY_TEST==True:
    with torch.no_grad():
        y_predicted = model(X_test)
        y_predicted_cls = y_predicted.round()
        acc = y_predicted_cls.eq(Y_test).sum()/float(Y_test.shape[0])
        print(f'accuracy: {acc.item():.4f}')
