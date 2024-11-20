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
import torchsummary

from torch.utils.data import Dataset

# Database loader
class DryingDataset(Dataset):

    # Initialization function
    def __init__(self, directory):
        # Initialize parent modules
        super().__init__()

        return

    # Return size of dataset
    def __len__(self):
        
        return

    # Return array of time series
    def __getitem__(self, indx):
    
        return

# Fully-Connected Neural Network
class PINN(nn.Module):
    # Initialization function
    def __init__(self):
        # Initialize parent modules
        super().__init__()
        # Define the forward pass of the network sequentially
        self.network = nn.Sequential(
            nn.LazyLinear(out_features=16, device=device),
            nn.ReLU(),
        )

    def forward(self, x):

        # Move tensor to device
        x = x.to(device)
        # Return model's response to tensor
        return self.network(x)

# Set seed for random numbers
seed = datetime.datetime.now().timestamp()
torch.manual_seed(seed)
random.seed(seed)

# Set device for PyTorch
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.device(device)
torch.set_default_device(device)

# Print which device is being used
print("\nPyTorch: Using device " + device, end="\n\n")

# Initialize random tensor
tensor = torch.rand((3,128,128))

# Initialize model
model = PINN()

# Print summary of the model
torchsummary.summary(model, tensor.shape)

# Optimizer hyperparameters
learning_rate =0.001

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

