import torch 
import torch.nn 
import torch.nn.functional as f 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset  
from torchmetrics import MeanAbsoluteError
from torchmetrics import R2Score
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
mean_absolute_error = MeanAbsoluteError() 
r2 = R2Score()

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

nn.eval() 

xtest = np.array(X_test) 
ytest = np.array(Y_test)

xtest = scaler.transform(xtest) 
X_test = torch.FloatTensor(xtest) 
Y_test = torch.FloatTensor(ytest)

with torch.no_grad(): 
    Y_pred = nn(X_test) 
    Y_pred = Y_pred.squeeze() 
    print(Y_pred.shape) 
    print(Y_test.shape) 
    
    r2_score = r2(Y_pred, Y_test)  
    mae_score = mean_absolute_error(Y_pred,Y_test)
    print(f"r2 nn: {r2_score}")
    print(f"mae nn: {mae_score}")
    #looked to be 88.9/89% r2 score
    # print(r2_score)


def nnPrediction(age, dst, smh, sth, slh, npd): 
    arr = np.array([age, dst, smh, sth, slh, npd]) 
    arr = scaler.transform([arr])
    arr = torch.FloatTensor(arr)  
    with torch.no_grad(): 
        result = nn(arr)
    return result.item() 

#pred function 



