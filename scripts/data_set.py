import numpy as np
from numpy import genfromtxt

data2d = np.genfromtxt('set2d.csv', delimiter=",")
_, idx = np.unique(data2d, return_index=True, axis=0)
new_data2d = data2d[np.sort(idx),:]

np.savetxt("set2d_u.csv", new_data2d, delimiter=",")

print np.shape(new_data2d)
# print new_data3d
# print new_data3d==(0,0)
# print np.where((new_data3d==(0,0)).all(axis=1))[0]