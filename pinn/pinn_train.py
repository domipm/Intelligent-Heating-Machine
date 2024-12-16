import  torch
import  torch.nn            as      nn

from    torch.utils.data    import  DataLoader

import  pinn_dataset
import  pinn_model

# Directory of train dataset
train_dir = "./sample_data/database/train/"

# Initialize train dataset
train_dataset = pinn_dataset.DryingDataset(train_dir)

# Initialize model
model = pinn_model.PINN()
# Set model to train mode
model.train()

# Optimizer hyperparameters
learning_rate = 0.01
# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training hyperparameters
epochs = 1000
# Physics loss weight
lambda_data = 15
lambda_phys = 1
lambda_init = 1

# Train all dataset over multiple epochs
for epoch in range(epochs):

    # For each file in dataset
    for k, (time, vals) in enumerate(train_dataset):

        # Reset gradients
        optimizer.zero_grad()

        # Obtain output from model based on full time domain
        output = model(time)

        # Compute data loss
        loss_data = torch.mean((output - vals)**2)

        # Compute first derivative temperature dT/dt
        dT = torch.autograd.grad(output[:,0], time, torch.ones_like(output[:,0]), create_graph=True)[0]
        dH = torch.autograd.grad(output[:,1], time, torch.ones_like(output[:,1]), create_graph=True)[0]
        # Residual of differential equations
        rest = (dT - model.talpha + model.tbeta * output[:,0] + model.tgamma *  time )
        resh = (dH - model.halpha + model.hbeta * output[:,1] + model.hgamma *  time )
        # Compute physics loss
        loss_phys = torch.mean(rest**2) + torch.mean(resh**2)

        # Initial conditions
        T0 = vals[0,0]
        H0 = vals[0,1]
        # Initial condition loss
        loss_init = torch.mean((output[0,0] - T0)**2) + torch.mean((output[0,1] - H0)**2)

        # Compute joint loss
        loss = lambda_data * loss_data + lambda_phys * loss_phys + lambda_init * loss_init

        # Backpropagate joint loss
        loss.backward()

        # Perform optimizers step
        optimizer.step()

    # Print epoch info
    print("Epoch ", epoch + 1, "Loss ", loss.item(), end="\r")

# Save model weights
torch.save(model.state_dict(), "./pinn_weights.pt")