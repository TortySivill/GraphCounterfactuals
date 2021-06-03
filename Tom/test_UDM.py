from UDM import UDM

import pandas as pd
import numpy as np

# df = pd.read_csv("Tom/table3.csv")
# ordinal = ["O" in c for c in df.columns]

df = pd.read_csv("Tom/adult.csv")

print(df.head())


# X = df.values

# udm = UDM(X, ordinal)

# mask = np.ones((X.shape[0], X.shape[0]), dtype=bool)
# mask[2,0] = 0
# mask[0,2] = 0
# mask[1,3] = 0

# print(udm(X, mask=mask, placeholder=np.inf))

# print(udm(X[2], X[3]))