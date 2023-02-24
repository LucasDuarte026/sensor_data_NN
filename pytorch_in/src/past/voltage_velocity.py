import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# selecionar os hyperparameters para melhor rede
input_size = 1
output_size = 1
hidden_layers = 2
hidden_size = 32
learning_rate = 0.01
batch_size = 32
num_epochs = 100
amount = 100
SAVE = False

# Carregamento dos dados para dentro da rede
voltage_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-21180.dat', sep="\s+")
velo_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-vel-21180.dat', sep="\s+")
data = voltage_fake.assign(
    velocity=velo_fake['velocity']).assign(time=velo_fake['time'])
data = data[['voltage', 'velocity']]
data = data.head(amount)

print(data.head(amount))

scaler = MinMaxScaler()
data[['voltage', 'velocity']] = scaler.fit_transform(
    data[['voltage', 'velocity']])
print(data)
# Dividir o conjunto de dados em validação, teste e treino para controle de performance e de qualidade
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(
    train_data, test_size=0.2, random_state=42)

# Normalizando o dados para melhor manipulação da rede neural
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)
val_data = scaler.transform(val_data)

# Converter em tensor data para o correto formato
train_data = torch.Tensor(train_data)
test_data = torch.Tensor(test_data)
val_data = torch.Tensor(val_data)


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



def main():
    print("\n\n____--____--____Code____--____--____\n\n")
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
        # print('Epoch: {}, Loss: {:.6f}'.format(epoch+1, loss.item()))

    # Evaluate the neural network on the test set
    net.eval()
    with torch.no_grad():
        inputs = test_data[:, :-1]
        labels = test_data[:, -1].view(-1, 1)
        outputs = net(inputs)
        test_loss = criterion(outputs, labels)
    print('Test Loss: {:.6f}'.format(test_loss.item()))

    # Denormalize the predicted and true labels
    outputs_used = np.zeros((outputs.detach().numpy().shape[0], 2))
    outputs_used[:, 1] = outputs[:,0]

    labels_used = np.zeros((labels.shape[0], 2))
    labels_used[:, 1] = labels[:,0]

    pred_labels = scaler.inverse_transform(outputs_used)
    true_labels = scaler.inverse_transform(labels_used)

    # Computar a acurácia do modelo
    test_accuracy = 1 - abs(true_labels - pred_labels).mean() / true_labels.mean()
    print('Test Accuracy: {:.2f}%'.format(test_accuracy*100))

    # Visualização dos dados
    
    print(scaler.inverse_transform((pd.DataFrame(outputs)).assign(velocity = data.velocity)))
    print(data.velocity)

    # Save the trained model
    torch.save(net.state_dict(), 'model.pth')
    if SAVE ==True:
        file =open('saved_net.txt','w',encoding='utf-8')

        file.write(str(net))
        file.close
        


if __name__ == "__main__":
    main()
    print('\nchanged\n',scaler.inverse_transform(data))
    