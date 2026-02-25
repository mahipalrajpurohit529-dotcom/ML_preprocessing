import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv(r"C:\codings\datasets\covid_toy - covid_toy.csv")
df = df.dropna()

print(df.head(2))

x = df. drop(columns = ['has_covid' ])
y = df['has_covid' ]

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
x_train['gender'] = lb.fit_transform(x_train['gender'])
x_train['cough'] = lb.fit_transform(x_train['cough'])
x_train['city'] = lb.fit_transform(x_train['city'])


print(x_train.head(2))

