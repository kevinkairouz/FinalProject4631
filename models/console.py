import analysis 
import histogram 
from neuralNetwork import nnPrediction 
import model as model 
import matplotlib.pyplot as plt  
import model as md

def main(): 
    analysis  
    histogram 
    
main() 


a = int(input("Enter Age: ")) 
daily_screen_hours = float(input("Enter Daily Screen Hours: ")) 
social_media_hours = float(input("Enter Social Media Hours: ")) 
study_hours = float(input("Enter Study Hours: "))
sleep_hours = float(input("Enter Sleep Hours: ")) 
notis = int(input("Enter Notifications: ")) 

#Gradient Boosting Model 
print("Gradient Boosting Prediction: ") 

print(md.make_predict(a, daily_screen_hours, social_media_hours, study_hours, sleep_hours, notis)) 


#Neural Network 
print("Neural Network Prediction: ")
print(nnPrediction(a, daily_screen_hours, social_media_hours, study_hours, sleep_hours, notis))

