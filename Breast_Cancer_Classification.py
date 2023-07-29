import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data_df=pd.read_csv("/content/breast-cancer.csv")

data_df.head()

data_df.shape

data_df.info()

data_df.isnull().sum()

data_df.describe()

data_df['diagnosis'].value_counts()

data_df.groupby('diagnosis').mean()

a=data_df.drop(columns='diagnosis',axis=1)
b=a.drop(columns='id',axis=1)
c=data_df['diagnosis']

b.shape

c.shape

a_train,a_test,b_train,b_test=train_test_split(b,c,test_size=0.2,random_state=2)

a_train.shape

a_test.shape

b_train.shape

b_test.shape

model=LogisticRegression()
model.fit(a_train,b_train)

a_train_prediction=model.predict(a_train)
a_train_score=accuracy_score(b_train,a_train_prediction)
a_train_score

a_test_prediction=model.predict(a_test)
a_test_score=accuracy_score(b_test,a_test_prediction)
a_test_score

input1=(11.89,18.35,77.32,432.2,0.09363,0.1154,0.06636,0.03142,0.1967,0.06314,0.2963,1.563,2.087,21.46,0.008872,0.04192,0.05946,0.01785,0.02793,0.004775,13.25,27.1,86.2,531.2,0.1405,0.3046,0.2806,0.1138,0.3397,0.08365)
input_to_array=np.asarray(input1)
reshaped=input_to_array.reshape(1,-1)
predict=model.predict(reshaped)
if predict[0]=='B':
  print("This is B type cancer")
else:
  print("This is M type cancer")

import pickle

filename='breast_cancer_classification.sav'
pickle.dump(model,open(filename,'wb'))

