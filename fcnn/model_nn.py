import  os
import  random

import  torch
import  torch.nn            as      nn
from    torch.utils.data    import  Dataset

import  matplotlib.pyplot   as      plt

import  numpy               as      np
import  pandas              as      pd

import  scipy.signal        as      signal
from    scipy               import  interpolate

class DryingDataset(Dataset):

    def __init__(self, directory):

        # Initialize parent class
        super().__init__()

        # Set working directory for dataset
        self.directory = directory

        # Get all file in current directory
        self.files = [file for file in os.listdir(self.directory)]

        return
    
    def __len__(self):
        # Return length of dataset (all files)
        return len(self.files)
    
    def __getitem__(self, index):

        # Open .csv file corresponding to index
        df = pd.read_csv(os.path.join(self.directory, self.files[index])).fillna(0)

        # Humidity arrays
        hdatax = []
        hdatay = []
        # Temperature arrays
        tdatax = []
        tdatay = []

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
                hdatax.append( getattr(row, "H_tick")/1000/60 ) # Tick values in minutes
                hdatay.append( getattr(row, "H_val") )
        # Find values of temperature
        for row in df.itertuples(index=False, name="Pandas"):
            row_htick = getattr(row, "T_tick")
            if row_htick > drying_start and row_htick < drying_end:
                tdatax.append( getattr(row, "T_tick")/1000/60 ) # Tick values in minutes
                tdatay.append( getattr(row, "T_val") )

        # Clean up data for temperature and humidity
        tdatax, tdatay = self.data_clean(tdatax, tdatay)
        hdatax, hdatay = self.data_clean(hdatax, hdatay)

        # Convert data to tensors
        tdatax_t = torch.from_numpy(np.array(tdatax, dtype=float)).type(dtype=torch.float32).view(-1,1)
        tdatay_t = torch.from_numpy(np.array(tdatay, dtype=float)).type(dtype=torch.float32).view(-1,1)
        # hdatax_t = torch.from_numpy(np.array(hdatax, dtype=float)).type(dtype=torch.float32).view(-1,1)
        hdatay_t = torch.from_numpy(np.array(hdatay, dtype=float)).type(dtype=torch.float32).view(-1,1)

        # Time tensor (for full time domain)
        time_t = tdatax_t.clone().detach().requires_grad_(True)

        # Concatenate temperature and humidity tensor data
        datay_t = torch.cat((tdatay_t, hdatay_t), dim=1)

        return time_t, datay_t
    
    # Function used to clean-up the data, apply smoothing, and interpolate to fixed length
    def data_clean(self, x, y, ignore = 10, out_len = 250, sg_window = 25, sg_order = 3):

        # Ignore first few datapoints
        x = np.array(x[ignore:])
        y = np.array(y[ignore:])
        # Remove any duplicates
        _, unique_indices = np.unique(x, return_index=True)
        # Use only unique indices for data
        x = x[unique_indices]
        y = y[unique_indices]
        # Normalize time to range (0,1) (or drying time)
        x -= np.min(x)
        # x /= np.max(x)
        # Interpolate data
        interp = interpolate.interp1d(x, y, kind = "linear")
        x = np.linspace(min(x), max(x), out_len)
        y = interp(x)
        # Apply Savitzky-Golay filtering to smooth data
        y = signal.savgol_filter(y, window_length = sg_window, polyorder = sg_order)

        return x, y
    
# Fully-Connected Neural Network
class FCNN(nn.Module):
    # Initialization function
    def __init__(self):
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
            nn.LazyLinear(out_features=2)
        )

    def forward(self, x):
        # Return model's response to tensor
        return self.network(x)
    




# Directory of train dataset
train_dir = "../measurements/database/train/"

# Initialize train dataset
train_dataset = DryingDataset(train_dir)
# train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

# Obtain array of all files
dataset = []
for n in range(len(train_dataset)):
    dataset.append(train_dataset[n])

# Initialize model
model = FCNN()
# Set model to train mode
model.train()

# Optimizer hyperparameters
learning_rate = 0.01
# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training hyperparameters
epochs = 100

# Cumulative train loss
loss_t = []

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
        loss = torch.mean((output - vals)**2)
        
        # Backpropagate joint loss
        loss.backward()

        # Perform optimizers step
        optimizer.step()

    # Append to loss lists
    loss_t.append(loss.item())

    # Print epoch info
    print("Epoch ", epoch + 1, "Loss ", loss.item(), end="\r")

# Save model weights
torch.save(model.state_dict(), "./output/fcnn_weights.pt")

# Plot evolution of all loss terms over time
plt.title(r"Training Loss $\mathcal{L}$")
plt.plot(range(epochs), loss_t, label=r"$\mathcal{L}$")
plt.legend()
plt.savefig("./output/pinn_trainloss.png", dpi=300)
plt.show()





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
test_dataset = DryingDataset(test_dir)

# Initialize model
model = FCNN()
# Load pre-trained weights
model.load_state_dict(torch.load("./output/fcnn_weights.pt", weights_only=True))
# Set model to train mode
model.train()

# Optimizer hyperparameters
learning_rate = 0.001
# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training hyperparameters (only fine-tune)
epochs = 100
# Physics loss weight
lambda_data = 15
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
    loss = torch.mean((output - vals)**2)

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
        loss = torch.mean((output - vals_sample)**2)

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