import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression  
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
import mysql.connector as sql  
import numpy as np

df = pd.read_csv("data/social.csv")
df = df.dropna()    
df = df.drop(columns=["addiction_level", "focus_score"])  
df["age"] = pd.to_numeric(df["age"], downcast="integer") 
df["notifications_per_day"] = pd.to_numeric(df["notifications_per_day"], downcast="integer")
X = df.drop(columns=["productivity_score"])
Y = df["productivity_score"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=42, test_size=0.3)
model = GradientBoostingRegressor(n_estimators=150,learning_rate=0.1) 
model.fit(X_train,Y_train) 


while True: 
    age = int(input("Enter Age")) 
    dst = float(input("Enter Daily Screen Time: "))
    smh = float(input("Enter Social Media Hours: ")) 
    sth = float(input("Enter Study Hours: ")) 
    sh = float(input("Enter Sleep Hours: "))
    npd = int(input("Enter Notifications Per Day: "))
    arr = np.array([age, dst, smh, sth, sh, npd]) 
    print(model.predict(arr)) 
    choice = str(input("Would you like to go again? (Y or N)")) 
    choice = choice.upper() 
    if choice == "Y": 
        continue 
    else:
        print("Have a good day, exiting program")
        break 

