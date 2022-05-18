import pandas as pd
import numpy as np
print("input")
datalist = [np.array(map(int, input().split())) for _ in range(263)]
df = pd.DataFrame(datalist, columns=list(map(str,np.arange(263))))
with pd.ExcelWriter("StudentNames.xlsx") as writer:
    df.to_excel(writer, sheet_name="StudentNames")
excel_filename = 'StudentNames.xlsx'