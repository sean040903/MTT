import numpy as np
arr1 = np.empty((0, 4), int)
arr2 = np.array([[11,21,1,1],[1,1,1,1]])
arr3 = np.array([[11,21,1,1]])
arr1 = np.append(arr1,arr2,axis=0)
print(arr1)
print(arr3)
arr1 = np.append(arr1,arr3,axis=0)
print(arr1)
print(np.sum(arr1,axis=0))
print(np.sum(arr1,axis=1))
print(np.delete(arr1,-1,axis=1))
print(np.delete(arr1,-1,axis=0))

a=np.array([1,2,3])
print(a[1].dtype)
if a[1].dtype == int:
    print('int32 is int')
if a.size==3:
    print(a.size)
print(np.delete(a, -1, 0))