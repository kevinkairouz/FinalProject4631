import pandas as pd 
import matplotlib.pyplot as plt  


df = pd.read_csv("../data/social.csv").sample(50) 

plt.scatter(df["sleep_hours"], df["productivity_score"])
plt.scatter(df["study_hours"], df["productivity_score"])


plt.show()
