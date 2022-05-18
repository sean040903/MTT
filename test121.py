import numpy as np
import time

def CGS(i, mat):
    cgs1 = []
    for j in range(len(mat)):
        if mat[j][i] == 1:
            cgs1.append(j)
    return cgs1

def CGS1(i,mat):
    array1 = np.array(mat)[:,i]
    return np.nonzero(array1)[0]

def CGS2(i,mat):
    arr1 = np.array(mat)[:,i]
    return np.nonzero(arr1 > 0)[0]

mat = [[1,1,1],[1,0,1],[0,0,0],[1,1,0],[0,0,1],[1,1,1]]

start1 = time.time()
print(CGS(0,mat),CGS(1,mat))
print("time1 :", time.time() - start1)

start2 = time.time()
print(CGS1(0,mat),CGS1(1,mat))
print("time2 :", time.time() - start2)

start3 = time.time()
print(CGS2(0,mat),CGS2(1,mat))
print("time3 :", time.time() - start3)

print(np.array(mat))
arr1=np.array(mat)
print(arr1[0][1])