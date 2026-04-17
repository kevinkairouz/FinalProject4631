import torch 
import torch.nn 
import torch.functional as f 
from sklearn.model_selection import train_test_split 
from torch.utils.data import DataLoader 


#Possible that pytorch nn (fully connected/linear nn is made for testing/implementation purpose) 

nn = torch.nn.Sequential(
    torch.nn.Linear(6, 8), 
    torch.nn.ReLU(), 
    
    
)
