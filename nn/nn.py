import torch 
import torch.nn 
import torch.nn.functional as f 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset  
import torchmetrics.regression.mae as mae 
import torchmetrics.regression.r2 as r 
import pandas as pd 
import numpy as np 


#Data Cleaning & Assigning X & Y labels 
df = pd.read_csv("../data/social.csv") 
df = df.dropna()    
df = df.drop(columns=["addiction_level", "focus_score"])  
df["age"] = pd.to_numeric(df["age"], downcast="integer") 
df["notifications_per_day"] = pd.to_numeric(df["notifications_per_day"], downcast="integer")
X = df.drop(columns=["productivity_score"])
Y = df["productivity_score"]

#Splitting X and Y testing & training data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=42, test_size=0.3) 


#NN Architecture
nn = torch.nn.Sequential(
    torch.nn.Linear(6,4), 
    torch.nn.ReLU(),  
    torch.nn.Linear(4, 4), 
    torch.nn.ReLU(),  
    torch.nn.Linear(4,4), 
    torch.nn.ReLU(),  
    torch.nn.Linear(4,1)
) 
#Placing Dataframes inside of numpy arrays & then scale the data to place inside of tensors for training
xt = np.array(X_train) 
yt = np.array(Y_train)
scaler = StandardScaler() 
xt = scaler.fit_transform(xt)
xt = torch.FloatTensor(xt) 
yt = torch.FloatTensor(yt) 


#Places scaled float tensors inside of TensorDataset which is like a dataframe and then placed in a dataloader iterable 
#so we can iterate over each record of X and Y respectively
td = TensorDataset(xt, yt) 
ds = DataLoader(td,batch_size=32)  


#create objects of loss functions and metrics used later for model comparison
loss_func = torch.nn.MSELoss()    
mean_absolute_error = mae.MeanAbsoluteError() 
r2 = r.R2Score()

optimizer = torch.optim.SGD(nn.parameters() ,lr=0.001)





num_epochs = 100

nn.train()

for epoch in range(num_epochs): 
    for x_data, y_data in ds:  
        optimizer.zero_grad() 
        output = nn(x_data)
        output = output.squeeze() 
        
        loss = loss_func(output, y_data) 
         
        loss.backward() 
        optimizer.step() 
    print(loss)


    print(f"Epoch{epoch}") 
print("training complete")  


age = int(input("Enter Age: ")) 
dst = float(input("Enter Daily Screen Time: "))
smh = float(input("Enter Social Media Hours: ")) 
sth = float(input("Enter Study Hours: ")) 
sh = float(input("Enter Sleep Hours: "))
npd = int(input("Enter Notifications Per Day: "))
arr = np.array([age, dst, smh, sth, sh, npd]) 

arr = scaler.transform(arr.reshape(1, -1))

user_tensor = torch.FloatTensor(arr)  

user_tensor = user_tensor
nn.eval()

with torch.no_grad(): 
    print(nn(user_tensor))

