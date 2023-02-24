
import numpy as np

test = np.empty([2,2])
print(test)
for i in range(100):
    test = np.append(test,[i,i*2])
print(test[0])