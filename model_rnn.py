# Recurrent Neural Network Model for Time-Series Forecasting

'''

TO-DO / IDEAS

    - The main idea is to use a RNN (LSTM) Time-Series Forecasting to predict future
      values of the parameters (temperature and humidity) based on the previous values.
      Usually, this is done for a single time series over a long period of time,
      rather than many different shorter series.

'''

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Directories where data is
fdir = "./sample_data/database/"
files = np.sort(os.listdir(fdir))

# Empty numpy array to store humidity values
datax = []
datay = []

# Value to store first tick of each file
first_tick = 0

# Obtain the relevant array to forecast (humidity values)
for file in files:

    # Open .csv file as pandas dataframe
    df = pd.read_csv(fdir + file)
    df = df.fillna(0)

    # Find tick values when drying starts and ends
    for row in df.itertuples(index=False, name="Pandas"):
        if getattr(row, "Phase_val") == "Drying":
            drying_start = getattr(row, "Phase_tick")
        if getattr(row, "Phase_val") == "Unloading":
            drying_end = getattr(row, "Phase_tick")
    # Find values of humidity
    for row in df.itertuples(index=False, name="Pandas"):
        row_htick = getattr(row, "H_tick")
        if row_htick > drying_start and row_htick < drying_end:
            datax.append( getattr(row, "H_tick") + first_tick )
            datay.append( getattr(row, "H_val") )

    first_tick += datax[0]

# Plot data with arange and with tick values (to compare)
fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(w=12, h=6)
ax[0].plot(datax, datay)
ax[1].plot(np.arange(len(datay)), datay)
#plt.show()
plt.close()

# Split data into train and test datasets
train_datay = datay[:835]
test_datay = datay[835:]
x = np.arange(len(datay))
train_x = x[:835]
test_x = x[835:]
plt.plot(train_x, train_datay, marker=".")
plt.plot(test_x, test_datay, marker=".")
#plt.show()
plt.close()

# Setup PyTorch device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_default_device(device)
torch.device(device)

# Setup random seed
seed = datetime.datetime.now().timestamp()
torch.manual_seed(seed)

# Convert train and test data to tensors
X_train = torch.from_numpy(train_x).float().to(device)
X_train = X_train.unsqueeze(-1)
y_train = torch.from_numpy(test_x).float().to(device)

# Define LSTM Model
class LSTM(nn.Module):

    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):

        super().__init__()

        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, device=device)
        self.linear = nn.Linear(hidden_layer_size, output_size, device=device)
    
    def forward(self, input_seq):
        input_seq.to(device)
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions
    
# Initialize model
model = LSTM(input_size=1, hidden_layer_size=50, output_size=1).to(device)
# Loss function
loss = nn.MSELoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Number of epochs to train for
epochs = 100

# Training loop
for epoch in range(epochs):
    
    # Set model to train mode
    model.train()
    # Reset gradients
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(X_train).to(device)

    # Compute the loss
    loss = loss(y_pred, y_train).to(device)

    # Backpropagation
    loss.backward()
    optimizer.step()

    print("Epoch {}/{}\tLoss {}".format(epoch, epochs, loss.item()))