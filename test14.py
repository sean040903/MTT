import numpy as np
"""print(np.append(np.array([2,3]),[2],axis=0))
print(np.append(np.array([2,3]),[2]))
print(np.append(np.array([[2,3]]),[[2,3]],axis=1))
print(np.append(np.array([[2,3]]),[[2,3]],axis=0))
print(np.append(np.empty((0,2), int),np.array([[2,3]]),axis=0))
print(len(np.append(np.empty((0,2), int),np.array([[2,3]]),axis=0)))
print(len(np.append(np.array([2,3]),[2],axis=0)))
print(np.sum([1,2,3]))
arr2=np.append(np.empty((0,2), int),np.array([[2,38012801921892810829810829018029102891802810828108]]))
print(arr2)
if type(arr2[1]) is int:
    print(1)
print(2 in arr2)
print(np.where(arr2==2)[0][0])
print([2,38012801921892810829810829018029102891802810828108] in arr2)
print(np.sum(np.append(np.array([[2,3]]),[[2,3]],axis=0),axis=0))
print(np.sum(np.append(np.array([[2,3]]),[[2,3]],axis=0),axis=1))
print(np.sum(np.sum(np.append(np.array([[2,3]]),[[2,3]],axis=0),axis=0)))
a=0
if not a>1 or not a==0:
    print('it all works!')
print(np.arange(10)[4:10:2])
print(np.where(np.append(np.array([[2,3]]),[[2,3]],axis=0)==2)[0])
print(np.array([[[2,1],[3,4]],[[2,1],[3,4]]]))
arr3=np.array([[[2,1],[3,4]],[[2,1],[3,4]]])
print(np.where(arr3==1))
print(arr3//2)
print(np.union1d(arr3,arr3//2))
print(2*np.add(arr3,arr3//2))
arr4 = np.zeros((0,2,4), int)
print(arr4.shape)
print(arr4)
print(np.array([[]]).shape)
print(np.empty((1,1,3), int))"""

"""
arr5 = np.empty((1,1,3), int)
print(np.append(arr5,[[[7, 8, 9]]], axis=0))
print(np.append(arr5,[[[7, 8, 9]]], axis=0).shape)"""
"""
arr1 = np.array([2,3,4,5])
print(arr1.reshape((1,-1)))
arr0=arr1.reshape((1,-1))
print(arr0.shape)
print(arr0.ndim)
print(np.append(np.array([arr1]),[[arr1]]))
arr2 = np.array([[2,3,4,5]])
print(np.where(4 == arr1)[0][0])
print(np.where(4 == arr1)[0][0])
print(np.where(arr1 == np.min(arr1))[0][0])
print(np.argmin(np.array([[6,7,8,9],[2,3,4,5]])))
a= np.array([[6,7,8,9],[2,3,4,5],[0,0,0,10]])
print(a[np.lexsort(a.T)])

arr6=np.empty(shape=(2,2))"""
"""arr7=arr6.reshape(1,-1)
print(arr7.shape)
print(arr7.ndim)
print(np.append(arr7,[[1,2]]))
print(arr7.shape)
print(arr7.ndim)"""
"""item = np.array([[5, 5],[6, 6]])
arr8 = np.append(arr6,item.reshape(2,2),axis=0)
print(arr8)
print(arr8.ndim)
print(arr6)
print(arr6.ndim)
"""
arr1 = np.empty((0, 4), int)
arr2 = np.array([[11,21,1,1],[1,1,1,1]])
arr3 = np.array([[11,21,1,1]])
arr1 = np.append(arr1,arr2,axis=0)
print(arr1)
arr1 = np.append(arr1,arr3,axis=0)
print(arr1)

arr4 = np.empty((0), int)
arr4 = np.append(arr4,[2,4])
print(arr4)
arr4 = np.append(arr4, [3])
print(arr4)
arr4 = np.append(arr4.reshape(1,-1), [[2,3,4]],axis=0)
print(arr4)
print(type(np.array([4])))
print(np.base_repr(4, 2))
x = np.arange(4)
arr = {'int' , bin(4)}
print(arr)
x = np.array([4])
print(np.array2string(x, formatter={'int':lambda x: bin(x)}))
print(np.binary_repr(4))
nbf = np.binary_repr(4)
print(len(nbf))
if nbf[0] == '1':
    print('it is string')
print(str(np.binary_repr(4)))
print(len(np.binary_repr(4)))
print(np.array(list(np.binary_repr(4)), dtype=int))

print(abs(0-1))
print(np.array(['ffe4e1', 'f7bbbb', 'ff7f50', 'ffb3d9', 'ffff00', '7fff00', 'd2b5fc', 'b4d0fd', 'afeeee', '00ffff', 'b8ffe4', 'd8bfd8', 'ffca99', 'ff69b4'])[10])

a = np.array([[1,2,3],[2,3,2]])


print(a[np.lexsort(a.T)])
a = a[np.lexsort(a.T)]
print(a)

print(np.sum(SL,axis=1))