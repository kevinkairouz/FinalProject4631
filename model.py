import pandas as pd  
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import VotingClassifier 


df = pd.read_csv("prompts.csv")  
#used to check for amount of null values in our Dataframe (there are zero) 
# print(df.isna().sum())

