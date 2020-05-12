# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:57:02 2020

@author: SInuganti
"""
def previousState(X):
     X= dataset.iloc[:,0:3].values
     imp.fit(X[:,1:3])
     X[:,1:3]=imp.fit_transform(X[:,1:3])
     print(X)
     return X
    

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('C:/Users/SInuganti/Desktop/Machine Learning/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Data.csv')
X= dataset.iloc[:,0:3].values
Y= dataset.iloc[:,-1:].values
from sklearn.impute import SimpleImputer

# imp=SimpleImputer(missing_values=np.nan, strategy='mean')
# imp.fit(X[:,1:3])
# X[:,1:3]=imp.fit_transform(X[:,1:3])
# X= dataset.iloc[:,0:3].values

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(random_state=0)
imp.fit(X[:,1:3])
X[:,1:3]=imp.fit_transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# LE= LabelEncoder()
# X[:,0]=LE.fit_transform(X[:,0])

# enc= OneHotEncoder()
# encoded_array=(enc.fit_transform(X[:,[0]]).toarray())

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('c',OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

#X=previousState(X)




