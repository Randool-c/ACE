import numpy as np
import numpy.matlib


a = np.arange(12).reshape(3, 4)
print(a)
b = np.arange(12).reshape(4, 3)
print(b)
c = np.matmul(a, b)
print(c)
a = {'hello': 1}
print('hello' in a);

# d = np.memmap('haha.dat', mode='w+', shape=(30000, 30000))
# a = np.matlib.zeros((30000, 30000), float)
