import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression  
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 
import mysql.connector as sql 

@dataclass 
class User: 
    age: int 
    daily_screen_time: float 
    study_hours: float 
    sleep_hours: float 
    noti_per_day: int 
    focus_score: float
    productivity_score: float
    

df = pd.read_csv("social.csv")
#print(df.isna().sum())
df = df.dropna()    
df = df.drop(columns=["addiction_level"])  
print(df.info())
df["age"] = pd.to_numeric(df["age"], downcast="integer") 
df["notifications_per_day"] = pd.to_numeric(df["notifications_per_day"], downcast="integer")
X = df.drop(columns=["productivity_score"])
Y = df["productivity_score"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=42, test_size=0.3)
# model = LinearRegression() 
model = GradientBoostingRegressor()
model.fit(X_train,Y_train) 
# print(df.info())
print(model.score(X_test,Y_test)) 
# benchamarked at 87.4%
#in stock testing 

def predict(age, dst, smh, sth, slh, npd): 
    # L = [u.age, u.daily_screen_time, u.study_hours, u.sleep_hours, u.noti_per_day, 
    #      u.addiction_high, u.addiction_mid, u.addicition_low] 
    udf = pd.DataFrame({"age": [age], "daily_screen_time": [dst], "social_media_hours": [smh], 
                        "study_hours": [sth], "sleep_hours": [slh], "notifications_per_day": [npd]}) 
     
    return model.predict(udf)

def sendPrediction():  
    return ""  


