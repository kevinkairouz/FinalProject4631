import pandas as pd 
import matplotlib.pyplot as plt  
import model as m
import neuralNetwork as neuralNetwork

import numpy as np

df = pd.read_csv("../data/social.csv").sample(20)  

fig, axis = plt.subplots(2, 2, layout="constrained") 


axis[0,0].scatter(df["sleep_hours"], df["productivity_score"], color = "orange")   
axis[0,0].set_title("Sleep Hours & Productivity") 
axis[0,0].set_xlabel("Hours of Sleep") 
axis[0,0].set_ylabel("Productivity")

axis[0,1].scatter(df["study_hours"], df["productivity_score"], color = "yellow") 
axis[0,1].set_title("Study Hours & Productivity")
axis[0,1].set_xlabel("Hours Studying") 
axis[0,1].set_ylabel("Productivity")



axis[1,0].scatter(df["daily_screen_time"], df["sleep_hours"], color = "purple") 
axis[1,0].set_title("Daily Screen Hours & Sleep Hours")
axis[1,0].set_xlabel("Daily Screen Hours") 
axis[1,0].set_ylabel("Hours of Sleep")


axis[1,1].scatter(df["social_media_hours"], df["productivity_score"], color = "brown")
axis[1,1].set_title("Social Media Hours & Productivity") 
axis[1,1].set_xlabel("Hours on Social Media") 
axis[1,1].set_ylabel("Productivity") 


plt.show()

plt.scatter(m.X_test["age"][0:25],m.Ypred[0:25], color = ["pink"], marker="*", linewidths=2.5, label = "Prediction") 
plt.scatter(m.X_test["age"][0:25],m.Y_test[0:25], color = ["black"], marker="P", linewidths=2.5, label = "Actual Y LABEL/SCORE")  
plt.title("Gradient Boosted Predictions vs Actual Y Values") 
plt.legend() 
plt.tight_layout()
plt.show()


nn_r2 = neuralNetwork.r2_score 
nn_mae = neuralNetwork.mae_score 

gb_r2 = m.r2 
gb_mae = m.mae 
 
fig, axis = plt.subplots(1,2) 

axis[0].bar([0, 1],[nn_r2, gb_r2], color = ["green", "orange"], label = ["Neural Network", "Gradient Boosting"])  
axis[0].set_title("R2 Score Comparison (Green: NN, Orange: GBoost)")



axis[1].bar([0, 1], [nn_mae,gb_mae], color = ["green", "orange"], label = ["Neural Network", "Gradient Boosting"])
axis[1].set_title("Mean Absolute Error Comparison (Green: NN, Orange: GBoost)")

plt.tight_layout() 
plt.show()  

#showing nn blueprint
img = plt.imread("../images_analysis/nn_image.png")  
plt.imshow(img) 
plt.axis("off") 
plt.show()   

Yprednp = neuralNetwork.Y_pred.numpy() 
Xtestnp = neuralNetwork.X_test.numpy() 

plt.title("Neural Network Predictions vs Actual Y Labels")
plt.scatter(Xtestnp[:20, 0],Yprednp[0:20],color = ["green"], label = ["Neural Network"], linewidths=3, marker="v") 
plt.scatter(Xtestnp[:20, 0], neuralNetwork.Y_test[0:20], color = ["red"], label = ["Y Labels"], linewidths=3, marker="s")
plt.legend()

plt.show()

bins = [0,18,25,30,40]

labels = ["Under 18", "18-25", "25-30", "30-40"] 
m.df["age_group"] = m.pd.cut(m.df["age"], bins=bins, labels=labels)


#average social media hours based on age group
fig, axis = plt.subplots(1, len(labels), figsize=(15, 5), sharey=True)
for i, group in enumerate(labels):
    subset = m.df[m.df["age_group"] == group]["social_media_hours"]
    axis[i].hist(subset, color="steelblue", edgecolor="black")
    axis[i].set_title(group)
    axis[i].set_xlabel("Social Media Hours")
    axis[i].set_ylabel("Count")
plt.suptitle("Social Media Hours by Age Group")
plt.tight_layout()
plt.show()

#comparing how much of a persons screen time is taken up by social media based on age group
grouped = m.df.groupby("age_group", observed=True)[["daily_screen_time", "social_media_hours"]].mean()
grouped["other_screen_time"] = grouped["daily_screen_time"] - grouped["social_media_hours"]

plt.figure()
plt.bar(grouped.index, grouped["social_media_hours"], label="Social Media", color="steelblue", edgecolor="black")
plt.bar(grouped.index, grouped["other_screen_time"], bottom=grouped["social_media_hours"], label="Other Screen Time", color="orange", edgecolor="black")
plt.title("Daily Screen Time vs Social Media Hours by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Hours")
plt.legend()
plt.tight_layout()
plt.show()