import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv(r"C:\codings\datasets\covid_toy - covid_toy.csv")
print(df.head())


# Handling null :- 

df = df.dropna()

# ENcoding:-
df_cat = df.select_dtypes(include='str').columns
lb = LabelEncoder()
for x in df_cat:
    df[x] = lb.fit_transform(df[x])


# Spliting :-
x = df.drop(columns = ['has_covid'])
y = df['has_covid']

x_train , x_test ,y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)

joblib.dump(model , 'lr_model.pkl')