import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


df = pd.read_csv(r"C:\codings\datasets\Social_Network_Ads - Social_Network_Ads.csv")

df =df.drop(columns=['User ID','Gender'])

x = df.drop(columns=['Purchased'])
y = df['Purchased']

x_train ,x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)

joblib.dump(model , 'lr_model.pkl')

print(df)