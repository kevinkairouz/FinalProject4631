import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv("../data/social.csv") 

plt.hist(df["age"], color="green")  
plt.title("Age Distribution")
plt.show()
