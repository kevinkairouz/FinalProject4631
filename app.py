from flask import Flask, request,jsonify
from flask_cors import CORS, cross_origin
import mysql.connector as sql 
import model as m 

app = Flask(__name__) 
CORS(app)


@app.route("/")
@cross_origin() 
def greeting(): 
    return "Welcome"

@app.route("/predict", methods = ["POST"])
@cross_origin()
def makePrediction(): 
    data = request.json() 
    user_age = data["age"] 
    user_daily_screen_time = data["daily_screen_time"] 
    user_social_media_hours = data["social_media_hours"] 
    user_study_hours = data["study_hours"] 
    user_sleep_hours = data["sleep_hours"] 
    user_notifications_per_day = data["notifications_per_day"] 

    pred = m.predict(user_age, user_daily_screen_time,user_social_media_hours,user_study_hours,user_sleep_hours,user_notifications_per_day)
    return jsonify({"productivity_score": pred}) 

app.run(debug=True)