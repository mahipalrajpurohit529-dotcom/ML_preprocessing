import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\codings\datasets\insurance - insurance.csv")




# print(df.columns)
# print(df.info())




x = df.drop(columns=['charges'])
y = df['charges']


x_train , x_test , y_train , y_test = train_test_split(x, y , test_size = 0.2,random_state=34)




# print(df.isna().sum())
# no null values
# print(df.nunique())




# a total of 7 columns

# encoding the catagorical once :- 
ohe = OneHotEncoder(drop = 'first' , sparse_output = False )
x_train_sex_smoker_region = ohe.fit_transform(x_train[['sex' , 'smoker','region']])
x_test_sex_smoker_region = ohe.fit_transform(x_test[['sex' , 'smoker','region']])
# print(x_train_sex_smoker_region.shape)




# extracting the numeric data :- 

x_train_age_bmi_children = x_train.drop(columns =['smoker', 'region','sex']).values
x_test_age_bmi_children = x_test.drop(columns =['smoker', 'region','sex']).values
# print(x_train_age_bmi_children.shape)


# murgung:-
x_train_transformed = np.concatenate((x_train_age_bmi_children ,x_train_sex_smoker_region) , axis = 1)
print(x_train_transformed.shape)
# shape is (1070,8)






transformers = ColumnTransformer(transformers=[
    ('tnf1',OneHotEncoder(sparse_output=False, drop='first'),['sex','smoker','region'])
],remainder='passthrough')

print(transformers.fit_transform(x_train).shape)