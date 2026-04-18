import torch 
import torch.nn 
import torch.functional as f 
from sklearn.model_selection import train_test_split 
from torch.utils.data import DataLoader, TensorDataset 
import pandas as pd 
import numpy as np 

df = pd.read_csv("../data/social.csv") 

df = df.dropna()    
df = df.drop(columns=["addiction_level", "focus_score"])  
df["age"] = pd.to_numeric(df["age"], downcast="integer") 
df["notifications_per_day"] = pd.to_numeric(df["notifications_per_day"], downcast="integer")
X = df.drop(columns=["productivity_score"])
Y = df["productivity_score"]


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=42, test_size=0.3) 



#Possible that pytorch nn (fully connected/linear nn is made for testing/implementation purpose) 

nn = torch.nn.Sequential(
    torch.nn.Linear(6,4), 
    torch.nn.Sigmoid(),  
    torch.nn.Linear(4, 4), 
    torch.nn.Sigmoid(), 
    torch.nn.Linear(4,4), 
    torch.nn.Sigmoid(), 
    torch.nn.Linear(4,1)
)