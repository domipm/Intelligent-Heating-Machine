# Physics-Informed Neural Network

'''

TO-DO / IDEAS:

    - Use Physics-Informed Neural Network to estimate Temperature and Humidity of the air.
    This is done by creating a simple fully-connected model with a loss function given
    by the coupled ordinary differential equations of our system. The unknown parameters
    will also be included as learnable parameters by the model. Focus only on drying for now.
    Inputs: Time, Temperature, Humidity
    Parameters: (relevant parameters in differential equations)
    Outputs: Time, Temperature, Humidity
    Once the model has learned the relevant parameters, we can continue training with rest of files.
    Lastly, we evaluate the model on new data an see how it performs. The goal is that, by only using
    a couple of experimental data points (first few in time series), we can estimate how the total graph
    will look like.

'''

# Import all necessary libraries

import os
import random
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Fully-Connected Neural Network
class PINN(nn.Module):
    # Initialization function
    def __init__(self, in_shape):
        # Initialize parent modules
        super().__init__()
        # Define the forward pass of the network sequentially
        self.network = nn.Sequential(
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
            nn.LazyLinear(out_features=in_shape)
        )
        # Trainable parameter alpha
        self.alpha = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        # Return model's response to tensor
        return self.network(x)

# Directories where data is
fdir = "./sample_data/database/"
files = np.sort(os.listdir(fdir))

# Load data from one file (eg. 1990)
datax = []
datay = []

# Open .csv file as pandas dataframe
df = pd.read_csv(fdir + "1990.csv")
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
        datax.append( getattr(row, "H_tick") )
        datay.append( getattr(row, "H_val") )

# Ignore first few datapoints (seem to be problematic)
datax = datax[10:]
datay = datay[10:]

# Extract some values to use as training points (take every n-th data point)
datax_train = datax[0::10]
datay_train = datay[0::10]
# Only consider first few points
#datax_train = datax_train[:8]
#datay_train = datay_train[:8]

# Convert to tensors
datax_train_t = torch.from_numpy(np.array(datax_train, dtype=float)).type(dtype=torch.float32).view(-1,1)
datay_train_t = torch.from_numpy(np.array(datay_train, dtype=float)).type(dtype=torch.float32).view(-1,1)

# Discretize time values over domain
datax_physics = torch.from_numpy(np.linspace(min(datax), max(datax), len(datax_train_t))).requires_grad_(True).type(dtype=torch.float32).view(-1,1)

# Plot measured data and training points
plt.plot(datax, datay, label="Measured Data")
plt.plot(datax_train, datay_train, ".", label="Sample Datapoints")

# Set seed for random numbers
seed = datetime.datetime.now().timestamp()
torch.manual_seed(seed)
random.seed(seed)

# Initialize model
model = PINN(in_shape = 1) # len(datax_train_t))

# Optimizer hyperparameters
learning_rate = 0.001
# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training hyperparameters
epochs = 100000
# Physics loss weight
alpha_data = 1
alpha_phys = 0

# Training loop
for epoch in range(epochs):

    # Reset gradients to zero
    optimizer.zero_grad()

    # Compute output from model using data samples
    out_data = model(datax_train_t)

    # Compute output from model using physics samples
    out_physics = model(datax_physics)

    # Compute data loss (mean squared error)
    loss_data = torch.mean((out_data - datay_train_t)**2)

    # First derivative (dy/dx)
    dy = torch.autograd.grad(out_physics, datax_physics, torch.ones_like(out_physics), create_graph=True)[0]
    # Residual of differential equation
    res = dy + model.alpha * out_physics
    # Compute physics loss
    loss_physics = torch.mean(res**2)

    # Compute joint loss
    loss = alpha_data * loss_data + alpha_phys * loss_physics

    # Backpropagate joint loss
    loss.backward()

    # Perform optimizers step
    optimizer.step()

    # Print epoch info
    print("Epoch ", epoch, "Loss ", loss.item(), "a = ", model.alpha.item())

plt.plot(datax_physics.detach().numpy(), out_physics.detach().numpy(), label="Neural Network")
plt.plot(datax_physics.detach().numpy(), np.exp(-model.alpha.detach().numpy()*datax_physics.detach().numpy()), label="Function Predicted")
plt.legend()
plt.show()