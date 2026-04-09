import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression  
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
import mysql.connector as sql  
import numpy as np

    

df = pd.read_csv("data/social.csv")
#print(df.isna().sum())
df = df.dropna()    
df = df.drop(columns=["addiction_level", "focus_score"])  
# print(df.info())
df["age"] = pd.to_numeric(df["age"], downcast="integer") 
df["notifications_per_day"] = pd.to_numeric(df["notifications_per_day"], downcast="integer")
X = df.drop(columns=["productivity_score"])
Y = df["productivity_score"]


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=42, test_size=0.3)
model = GradientBoostingRegressor(n_estimators=150,learning_rate=0.1) 
model.fit(X_train,Y_train) 

# print(model.score(X_test,Y_test))  

# benchamarked at 87.4%
#in stock testing 

def predict(age, dst, smh, sth, slh, npd): 
    a = np.array([age, dst, smh, sth, slh, npd])  
    res = model.predict([a]) 
    sendPrediction(age, dst, smh, sth, slh, npd, res[0])
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

# for testing purpose 
# sendPrediction(25, 2.4, 4.5, 5.9, 4.0, 200, 70.5) 
predict(25,3,5,7,7,93) 


#2: Have a histogram for the ages to show diversity within the data ie meaning we can 
# defend our work  

""" 
If we have solid diversity within our data we prove not only is our data of such high quality, 
but we are able to show that it is indeed reliable data that is free of bias, therefore data quality 
is not a concern for our model 

""" 


#3: Create a bar chart that indicate what our averages are in the data, ie what is the avg for 
#age, social media time, and we can go further to say for each age range what is the average time 

