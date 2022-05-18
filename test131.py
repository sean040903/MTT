import numpy as np

def CGS(i,mat):
    return np.nonzero(np.array(mat)[:,i])[0]

print(CGS())
