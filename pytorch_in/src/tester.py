import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

amount = 120000
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
test = []
for i in range(len(data)):
    # print(f' | {data.loc[i,"voltage"]} | {data.loc[i,"voltage"]}')
    test.append(i)
print(test)
    
# Create scatter plot of data points
plt.scatter(data['voltage'], data['velocity'], c=data['velocity'], cmap='viridis', label='')

# Add predicted values to plot
# plt.scatter(data['voltage'], predictions.squeeze(), c='red', label='Predicted')

# Add axis labels and legend
plt.xlabel('Voltage')
plt.ylabel('Velocity')
plt.legend("viva la vida")

# Show plot
plt.show()