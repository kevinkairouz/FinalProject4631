import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 


@dataclass 
class User: 
    age: int


df = pd.read_csv("social.csv")
#print(df.isna().sum())
df = df.dropna()   
df = pd.get_dummies(df, columns=["addiction_level"]) 
df["age"] = pd.to_numeric(df["age"], downcast="integer") 

print(df.info()) 
print(df.sample().to_string())


X = df.drop(columns=["productivity_score"])
Y = df[["productivity_score", "focus_score"]] 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=42)
model = LinearRegression()
model.fit(X_train,Y_train)
print(model.score(X_test,Y_test)) 
# benchamarked at 87.5%
#in stock testing 

