import pandas as pd 
import matplotlib.pyplot as plt  


df = pd.read_csv("../data/social.csv").sample(50)  

fig, axis = plt.subplots(2, 2) 

axis[0,0].scatter(df["sleep_hours"], df["productivity_score"]) 

plt.show()

# Correlation of sleeping and productivity 
# Correlation of studying and productivity
# plt.scatter(df["sleep_hours"], df["productivity_score"])
# plt.scatter(df["study_hours"], df["productivity_score"])


# plt.show()
