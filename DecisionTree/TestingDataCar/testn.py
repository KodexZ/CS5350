import numpy as np
from sklearn.preprocessing import normalize

arr = np.array([[2, 2, 2]])



vec = normalize(arr, norm='l1')
print(1 - np.dot(vec, np.transpose(vec))[0][0])