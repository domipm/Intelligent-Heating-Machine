import  torch

import  matplotlib.pyplot   as      plt

import  pinn_dataset
import  pinn_model

# Directory of train dataset
test_dir = "../sample_data/database/test/"

# Initialize train dataset
train_dataset = pinn_dataset.DryingDataset(test_dir)

# Initialize model
model = pinn_model.PINN()
# Load pre-trained weights
model.load_state_dict(torch.load("./output/pinn_weights.pt", weights_only=True))
# Set model to train mode
model.train()

# Optimizer hyperparameters
learning_rate = 0.001
# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training hyperparameters (only fine-tune)
epochs = 100
# Physics loss weight
lambda_data = 0.5
lambda_phys = 0.5
lambda_init = 0.5

# Choose one file to fine-tune and evaluate
time, vals = train_dataset[0]

# Train all dataset over multiple epochs
for epoch in range(epochs):

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

plt.title("Drying Process PINN")
plt.plot(time.detach().numpy(), vals[:,0].detach().numpy(), color="blue", linestyle="--")
plt.plot(time.detach().numpy(), vals[:,1].detach().numpy(), color="red", linestyle="--")
plt.plot(time.detach().numpy(), output[:,0].detach().numpy(), color="blue", label="Temperature")
plt.plot(time.detach().numpy(), output[:,1].detach().numpy(), color="red", label="Humidity")
plt.legend()
plt.savefig("./output/pinn_graph.png", dpi=300)
plt.show()