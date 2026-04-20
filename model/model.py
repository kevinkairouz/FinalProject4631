import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import mysql.connector as sql  
import numpy as np

    
# first file path is used for the api when backend is activated/server is up and running 
# df = pd.read_csv("data/social.csv") 


df = pd.read_csv("../data/social.csv")
#print(df.isna().sum())
df = df.dropna()    
df = df.drop(columns=["addiction_level", "focus_score"])  
# print(df.info())
df["age"] = pd.to_numeric(df["age"], downcast="integer") 
df["notifications_per_day"] = pd.to_numeric(df["notifications_per_day"], downcast="integer")
X = df.drop(columns=["productivity_score"]) 
Y = df["productivity_score"]

scaler = StandardScaler()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=42, test_size=0.3) 
model = GradientBoostingRegressor(n_estimators=150,learning_rate=0.1) 
model.fit(X_train,Y_train)  

Ypred = model.predict(X_test) 

mae = mean_absolute_error(Y_test,Ypred)
# accuracy/r2 is benchamarked at 88.2%
r2 = model.score(X_test,Y_test)  



def predict(age, dst, smh, sth, slh, npd): 
    a = np.array([age, dst, smh, sth, slh, npd])  
    res = model.predict([a]) 
    # sendPrediction(age, dst, smh, sth, slh, npd, res[0])
    return res[0]

def sendPrediction(age, dst, smh, sth, slh, npd, ps): 
    db = sql.connect(host = "localhost", user = "root", password = "Dominics1", database ="finalProj") 
    query = "insert into user_predictions (age, daily_screen_time, social_media_hours,study_hours,sleep_hours,notifications_per_day,productivity_score) values (%s,%s,%s,%s,%s,%s,%s)"
    data = (age, dst, smh, sth, slh, npd, ps)
    cursor = db.cursor() 
    cursor.execute(query,data) 
    db.commit() 
    db.close()
    return "Success"  

