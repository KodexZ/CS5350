from typing import DefaultDict
from numpy import recarray
from pandas.core.frame import DataFrame
from scipy.stats import entropy
import pandas as pd
from dataclasses import dataclass
from typing import List
import numpy as np
import sys
from sklearn.preprocessing import normalize
from pandas.api.types import is_numeric_dtype
    
    
import pandas as pd
import numpy as np
print(np.exp(1))
print(np.log(np.exp(1)))
df = pd.DataFrame({'Date' : ['11/8/2011', '11/9/2011', '11/10/2011',
                                        '11/11/2011', '11/12/2011'],
                'Event' : [3, 4, 2, 1, 2]})
df.loc[(df['Event'] == 'Painting'),'Event']='Hip-Hop'
df.at[2, 'Event']+=2
df.loc[len(df)] = df.loc[0]
df = df.drop(df.columns[[0]], 1)
print(df)