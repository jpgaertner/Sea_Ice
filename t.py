import numpy  as np
import matplotlib.pyplot as plt

a = np.arange(9)
a = np.reshape(a, (3,3))

print(a)

b = np.sum(a[:,:], axis=0)

print(b)


