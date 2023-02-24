import numpy as np
import pandas as pd

df = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/neural_network_2023/graph_images/grandão/dados_criados.dat', sep="\s+")
diff = (df.predicted - df.original)
diff_media = diff.abs().mean()
diff_max=diff.abs().max()
diff_min=diff.abs().min()
print(f'media da diferença: {diff_media:6.6f}\nmáxima diferença:    {diff_max:6.6f}\nMínima diferença:   {diff_min:6.6f}')
