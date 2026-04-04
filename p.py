import pandas as pd 

#this file is for testing and seeing values within csv 

df = pd.read_csv("social.csv") 

print(df["focus_score"].sample()) 
print(df["productivity_score"].sample()) 

