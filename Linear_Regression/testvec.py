import numpy as np
import pandas as pd
v1 = np.array([1, 1, 1, 1])

print(np.dot(v1, v1))

df = pd.DataFrame({'T' : [1, 2, 3, 4]})

df.apply(lambda row : print(row[:-1].to_numpy()))