# Recurrent Neural Network Model for Time-Series Forecasting

'''

TO-DO / IDEAS

    - The main idea is to use a RNN Time-Series Forecasting to predict future values
      of the parameters (temperature and humidity) based on the previous values.
      Usually, this is done for a single time series over a long period of time,
      rather than many different shorter series. Concatenation doesn't make sense!

'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

directory = "./sample_data/database/train/"
files = np.sort(os.listdir(directory))

# Concatenate all the data available
T = []
TH = []

H = np.empty(0)

for file in files:

    df = pd.read_csv(directory + file)
    df = df.fillna(0)

    for row in df.itertuples(index=False, name="Pandas"):
        if getattr(row, "H_tick") > df["Dry_tick"].iloc[0]:
            T.append(getattr(row, "T_val"))
            TH.append(getattr(row, "TH_val"))
            #H.append(getattr(row, "H_val"))
            H = np.append(H, getattr(row, "H_val"))

# Already padded! Same length!
Tx = np.arange(start=0, stop=len(T))
THx = np.arange(start=0, stop=len(TH))
Hx = np.arange(start=0, stop=len(H))
print(len(Tx), len(THx), len(Hx))

# For now, let's focus only on Humidity

# Create relevant pytorch tensors
x = torch.arange(0,len(H),1)
print(type(x))
# Convert numpy array into pytorch tensor
y = torch.from_numpy(H).to(torch.float32)

# Split data into train/test
test_size = 40
train_set = y[:-test_size]
test_set = y[:-test_size]

# Create data batches
def input_data(seq,ws):
    out = []
    L = len(seq)
    
    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window,label))
    
    return out

window_size = 40
train_data = input_data(train_set, window_size)

# Define model (LSTM)
class LSTM(nn.Module):

    def __init__(self, input_size = 1, hidden_size = 50, out_size = 1):

        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size,out_size)
        self.hidden = (torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))

    def forward(self,seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq),1,-1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq),-1))
        return pred[-1]
    
torch.manual_seed(42)
model = LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 10
future = 40

for i in range(epochs):

    print("Epoch " + str(i))
    
    for k, (seq, y_train) in enumerate(train_data):

        print("Batch " + str(k))

        optimizer.zero_grad()
        model.hidden = (torch.zeros(1,1,model.hidden_size),
                       torch.zeros(1,1,model.hidden_size))
        
        y_pred = model(seq)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {i} Loss: {loss.item()}")
    
    preds = train_set[-window_size:].tolist()
    for f in range(future):
        seq = torch.FloatTensor(preds[-window_size:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1,1,model.hidden_size),
                           torch.zeros(1,1,model.hidden_size))
            preds.append(model(seq).item())
        
    loss = criterion(torch.tensor(preds[-window_size:]), y[760:])
    print(f"Performance on test range: {loss}")

