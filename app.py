from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import mysql.connector as sql 
import model.model as m 

app = Flask(__name__) 
CORS(app)


@app.route("/")
@cross_origin() 
def greeting(): 
    return "Welcome"

@app.route("/predictGB", methods = ["POST"])
@cross_origin()
def makePrediction(): 
    data = request.json
    user_age = data["age"] 
    user_daily_screen_time = data["ScreenTime"] 
    user_social_media_hours = data["SocialHours"] 
    user_study_hours = data["StudyHours"] 
    user_sleep_hours = data["SleepHours"] 
    user_notifications_per_day = data["Noti"]  

    user_age = int(user_age) 
    user_daily_screen_time = float(user_daily_screen_time) 
    user_social_media_hours = float(user_social_media_hours)
    user_study_hours = float(user_study_hours)
    user_sleep_hours = float(user_sleep_hours)
    user_notifications_per_day = int(user_notifications_per_day) 
    
    pred = m.predict(user_age, user_daily_screen_time,user_social_media_hours,user_study_hours,user_sleep_hours,user_notifications_per_day)
    return jsonify({"productivity_score": pred})

@app.route("/predictNN", methods = ["POST"]) 
@cross_origin 
def predictNN(): 
    return None  

@app.route("/test", methods = ["GET"]) 
def send5(): 
    return jsonify({"Number": 5}) 


app.run()