import numpy as np

def lenprint(arr):
    return np.sum(np.floor(np.log10(np.where(arr == 0, 1, arr)))+1)

print(lenprint(np.array([1,2])))