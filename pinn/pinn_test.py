import  torch

import  matplotlib.pyplot   as      plt

import  numpy               as      np

import  pinn_dataset
import  pinn_model

# Root Mean Squared Error (RMSE)
def rmse(pred, vals):
    err = 0
    for i in range(len(pred)):
        err += (pred[i] - vals[i])**2
    err /= len(pred)
    err = np.sqrt(err)
    return err

# Directory of test dataset
test_dir = "../measurements/database/test/"

# Initialize test dataset
test_dataset = pinn_dataset.DryingDataset(test_dir)

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
lambda_data = 5
lambda_phys = 1
lambda_init = 1

# Choose one file to fine-tune and evaluate
index = np.random.randint(0, len(test_dataset))
time_full, vals_full = test_dataset[index]

# Use only first few points for fine-tuning
time = time_full[0:35:5]
vals = vals_full[0:35:5]

# Plot sampling points
plt.plot(time.detach().numpy(), vals[:,0].detach().numpy(), linestyle="", marker=".", color="blue")
plt.plot(time.detach().numpy(), vals[:,1].detach().numpy(), linestyle="", marker=".", color="red")

# Train file over multiple epochs
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
print()

# Make prediction based on the rest of the time series
model.eval()
with torch.no_grad():

    output = model(time_full)

# Compute Root Mean Squared Error (RMSE)
out_t = output.detach().numpy()[:,0]
out_h = output.detach().numpy()[:,1]
rmse_t = rmse(out_t, vals_full.detach().numpy()[:,0])
rmse_h = rmse(out_h, vals_full.detach().numpy()[:,1])

plt.title("Drying Process PINN Drying Batch {}".format(index))
# Real data
plt.plot(time_full.detach().numpy(), vals_full[:,0].detach().numpy(), color="blue", linestyle="--")
plt.plot(time_full.detach().numpy(), vals_full[:,1].detach().numpy(), color="red", linestyle="--")
# Outputs
plt.plot(time_full.squeeze(1).detach().numpy(), output[:,0].detach().numpy(), color="blue", label="Temperature")
plt.plot(time_full.squeeze(1).detach().numpy(), output[:,1].detach().numpy(), color="red", label="Humidity")
#plt.text(x=-0.25, y=0, s=r"RMSE$_T$ = {0:.2f}".format(rmse_t))
#plt.text(x=-0.25, y=-5.5, s=r"RMSE$_H$ = {0:.2f}".format(rmse_h))
plt.legend()
plt.savefig("./output/pinn_graph.pdf")
plt.show()

# Print RMSE calculated
print("\nRMSE (T) = ", (rmse_t).item(), "%")
print("\nRMSE (H) = ", (rmse_h).item(), "%")
print()

# Combined, average loss
loss_avg = 0

# Average RMSE
rmse_t_avg = 0
rmse_h_avg = 0

# Go over all files and evaluate, calculating the observed loss
for k, (time, vals) in enumerate(test_dataset):

    model.train()

    # Use only first few points for fine tuning
    time_sample = time[0:35:5]
    vals_sample = vals[0:35:5]

    # Train all dataset over multiple epochs
    for epoch in range(epochs):

        print("Computing average loss. File {} Epoch {}".format(k, epoch), end="\r")

        # Reset gradients
        optimizer.zero_grad()

        # Obtain output from model based on full time domain
        output = model(time_sample)

        # Compute data loss
        loss_data = torch.mean((output - vals_sample)**2)

        # Compute first derivative temperature dT/dt
        dT = torch.autograd.grad(output[:,0], time_sample, torch.ones_like(output[:,0]), create_graph=True)[0]
        dH = torch.autograd.grad(output[:,1], time_sample, torch.ones_like(output[:,1]), create_graph=True)[0]
        # Residual of differential equations
        rest = (dT - model.talpha + model.tbeta * output[:,0] + model.tgamma *  time_sample )
        resh = (dH - model.halpha + model.hbeta * output[:,1] + model.hgamma *  time_sample )
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

    # Make prediction based on the rest of the time series
    model.eval()
    with torch.no_grad():

        output = model(time_full)

    # Compute final loss
    loss_avg += torch.mean((output - vals)**2)

    # Compute RMSE
    rmse_t_avg += rmse(output.detach().numpy()[:,0], vals.detach().numpy()[:,0])
    rmse_h_avg += rmse(output.detach().numpy()[:,1], vals.detach().numpy()[:,1])

# Print average loss obtained
print()
print("\nAverage test loss = ", (loss_avg / len(test_dataset)).item())
# Print average RMSE obtained
print("\nAverage RMSE (T) = ", (rmse_t_avg / len(test_dataset)).item(), "%")
print("\nAverage RMSE (H) = ", (rmse_h_avg / len(test_dataset)).item(), "%")
print()