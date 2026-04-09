import pandas as pd 
import matplotlib.pyplot as plt  


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
