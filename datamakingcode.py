import openpyxl
import numpy as np
import pandas as pd
from openpyxl import load_workbook
l=62
"""
arr1=np.arange(l)
print(arr1)
totaldata=[]
for i in range(s):
    arr=np.random.permutation(arr1)
    totaldata.append(arr.tolist())
print(totaldata)
"""
arr=np.random.rand(l,l)
df=pd.DataFrame(arr.tolist(),columns=list(map(str,np.arange(l).tolist())))
with pd.ExcelWriter("DI.xlsx") as writer:
    df.to_excel(writer, sheet_name="Sheet1")


