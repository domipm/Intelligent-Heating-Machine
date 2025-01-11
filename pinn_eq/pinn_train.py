import  torch

import  pinn_dataset
import  pinn_model

import  random

import  numpy               as      np

import  matplotlib.pyplot   as      plt

from    torchsummary        import  summary

# Directory of train dataset
train_dir = "../measurements/database/train/"

# Initialize train dataset
train_dataset = pinn_dataset.DryingDataset(train_dir)

# Obtain array of all files
dataset = []
for n in range(len(train_dataset)):
    dataset.append(train_dataset[n])

# Initialize model
model = pinn_model.PINN()
# Set model to train mode
model.train()

# Print summary of the model
time, _ = train_dataset[0]
summary(model, time.shape)

# Optimizer hyperparameters
learning_rate = 0.01
# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training hyperparameters
epochs = 50
# Physics loss weight
lambda_data = 10
lambda_phys = 1
lambda_init = 0.25

# Cumulative train loss
loss_t = np.empty(shape=(epochs, 4), dtype=object)

# Train all dataset over multiple epochs
for epoch in range(epochs):

    # Shuffle files
    random.shuffle(dataset)

    # For each file in dataset
    for k, (time, vals) in enumerate(dataset):

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
        res_T = (dT + model.alpha_T + model.alpha_T_T * output[:,0] + model.alpha_T_dH * dH)
        res_H = (dH + model.alpha_H + model.alpha_H_H * output[:,1])
        # Compute physics loss
        loss_phys = torch.mean(res_T**2) + torch.mean(res_H**2)

        # Initial conditions
        T0 = vals[0,0]
        H0 = vals[0,1]
        # Initial condition loss
        loss_init = (output[0,0] - T0)**2 + (output[0,1] - H0)**2

        # Compute joint loss
        loss = lambda_data * loss_data + lambda_phys * loss_phys + lambda_init * loss_init       

        # Backpropagate joint loss
        loss.backward()

        # Perform optimizers step
        optimizer.step()

    # Append to loss lists
    loss_t[epoch] = [ loss.item(), loss_phys.item(), loss_data.item(), loss_init.item() ]

    # Print epoch info
    print("Epoch ", epoch + 1, "Loss ", loss.item(), end="\r")

# Save model weights
torch.save(model.state_dict(), "./output/pinn_weights.pt")

print("dT = {} + {}*T + {}*H + {}*t + {}*dH".format(model.alpha_T.item(), model.alpha_T_T.item(), model.alpha_T_H.item(), model.alpha_T_t.item(), model.alpha_T_dH.item()))
print("dH = {} + {}*T + {}*H + {}*t + {}*dH".format(model.alpha_H.item(), model.alpha_H_T.item(), model.alpha_H_H.item(), model.alpha_H_t.item(), model.alpha_H_dT.item()))

# Plot evolution of all loss terms over time
plt.title(r"Training Loss $\mathcal{L}$")
plt.plot(range(epochs), loss_t[:,1], label=r"$\mathcal{L}_{\text{physics}}$")
plt.plot(range(epochs), loss_t[:,2], label=r"$\mathcal{L}_{\text{data}}$")
plt.plot(range(epochs), loss_t[:,3], label=r"$\mathcal{L}_{\text{init. c.}}$")
plt.plot(range(epochs), loss_t[:,0], label=r"$\mathcal{L}_{\text{total}}$")
plt.legend()
plt.savefig("./output/pinn_trainloss.pdf")
plt.show()