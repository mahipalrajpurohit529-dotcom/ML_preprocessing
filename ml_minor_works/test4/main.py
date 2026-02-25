import pandas as pd 
import numpy as np 

df = pd.read_csv(r"C:\codings\datasets\covid_toy - covid_toy.csv")
print(df.head())


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler




x = df.drop(columns=['fever'])
y = df['fever']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=34)





cat_features = x_train.select_dtypes(include=['object']).columns
num_features = x_train.select_dtypes(exclude=['object']).columns



numerical_trns = Pipeline(steps=[
    ('imputing',SimpleImputer(strategy='mean')),
    ('scaling' , StandardScaler())
])

catagorical_trns = Pipeline(steps=[
    ('imputing',SimpleImputer(strategy='most_frequent')),
    ('encoding' , OneHotEncoder(handle_unknown='ignore'))
])


processor = ColumnTransformer(transformers=[
    ('trn1',numerical_trns,num_features),
    ('trn2',catagorical_trns,cat_features)
])