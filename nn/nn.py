import torch 
import torch.nn 
import torch.functional as f 
from sklearn.model_selection import train_test_split 
from torch.utils.data import DataLoader, TensorDataset 
import pandas as pd 
import numpy as np 

df = pd.read_csv("../data/social.csv") 



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