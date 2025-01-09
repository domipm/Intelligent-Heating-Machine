import  os
import  random
import  datetime

import  numpy               as      np
import  pandas              as      pd
import  matplotlib.pyplot   as      plt

import  scipy.signal        as      signal
from    scipy               import  interpolate

import  torch
import  torch.nn            as      nn

# Directories where data is
fdir = "./sample_data/database/"
files = np.sort(os.listdir(fdir))

# Humidity arrays
hdatax = []
hdatay = []
# Temperature arrays
tdatax = []
tdatay = []

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
        hdatax.append( getattr(row, "H_tick")/1000/60 ) # Tick values in minutes
        hdatay.append( getattr(row, "H_val") )
# Find values of temperature
for row in df.itertuples(index=False, name="Pandas"):
    row_htick = getattr(row, "T_tick")
    if row_htick > drying_start and row_htick < drying_end:
        tdatax.append( getattr(row, "T_tick")/1000/60 ) # Tick values in minutes
        tdatay.append( getattr(row, "T_val") )

# Function used to clean-up the data, apply smoothing, and interpolate to fixed length
def data_clean(x, y, ignore = 10, out_len = 250, sg_window = 25, sg_order = 3):

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

# Obtain cleaned-up data for temperature and humidity
tdatax, tdatay = data_clean(tdatax, tdatay)
hdatax, hdatay = data_clean(hdatax, hdatay)

# Data as tensors
tdatax_t = torch.from_numpy(np.array(tdatax, dtype=float)).type(dtype=torch.float32).view(-1,1)
tdatay_t = torch.from_numpy(np.array(tdatay, dtype=float)).type(dtype=torch.float32).view(-1,1)
hdatax_t = torch.from_numpy(np.array(hdatax, dtype=float)).type(dtype=torch.float32).view(-1,1)
hdatay_t = torch.from_numpy(np.array(hdatay, dtype=float)).type(dtype=torch.float32).view(-1,1)

# Datapoints to train (first few minutes)
sample_start = 0
sample_end = 35
sample_step = 5
tdatax_sample = tdatax[sample_start:sample_end:sample_step]
tdatay_sample = tdatay[sample_start:sample_end:sample_step]
hdatax_sample = hdatax[sample_start:sample_end:sample_step]
hdatay_sample = hdatay[sample_start:sample_end:sample_step]

# Datapoints as tensors
tdatax_sample_t = torch.from_numpy(np.array(tdatax_sample, dtype=float)).type(dtype=torch.float32).view(-1,1)
tdatay_sample_t = torch.from_numpy(np.array(tdatay_sample, dtype=float)).type(dtype=torch.float32).view(-1,1)
hdatax_sample_t = torch.from_numpy(np.array(hdatay_sample, dtype=float)).type(dtype=torch.float32).view(-1,1)
hdatay_sample_t = torch.from_numpy(np.array(hdatay_sample, dtype=float)).type(dtype=torch.float32).view(-1,1)

# Time tensor (for full time domain)
time_t = tdatax_t.clone().detach().requires_grad_(True)
time_sample_t = tdatax_sample_t.clone().detach().requires_grad_(True)

# Concatenate temperature and humidity tensor data samples
datay_t = torch.cat((tdatay_t, hdatay_t), dim=1)
datay_sample_t = torch.cat((tdatay_sample_t, hdatay_sample_t), dim=1)

# Plot sample data points
# plt.plot(tdatax_sample, tdatay_sample, color="tab:blue", markersize=8, linestyle="", marker=".")
# plt.plot(hdatax_sample, hdatay_sample, color="tab:orange", markersize=8, linestyle="", marker=".")

# Plot time series
plt.title("Drying Process Evolution")
plt.xlabel(r"Time $[\text{min}]$")
plt.ylabel("Sensor value")
plt.plot(tdatax, tdatay, label = "Temperature", color="tab:blue")
plt.plot(hdatax, hdatay, label = "Humidity", color="tab:orange")

# Set seed for random numbers
seed = datetime.datetime.now().timestamp()
torch.manual_seed(seed)
random.seed(seed)

# Fully-Connected Neural Network
class PINN(nn.Module):
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

        # Trainable parameters used in equations (constants approx.)
        self.talpha = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.tbeta = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.tgamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.tdelta = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.halpha = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.hbeta = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.hgamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)

        # Trainable array parameters
        self.water_content = nn.Parameter(torch.randn(len(time_t), dtype=torch.float32), requires_grad=True)
        self.surf_temperature = nn.Parameter(torch.randn(len(time_t), dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        # Return model's response to tensor
        return self.network(x)

# Initialize model
model = PINN()

# Optimizer hyperparameters
learning_rate = 0.01
# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training hyperparameters
epochs = 10000
# Physics loss weight
lambda_data = 15
lambda_phys = 1
lambda_init = 1

# Set model to train mode
model.train()

# Training loop
for epoch in range(epochs):

        # Reset gradients to zero
        optimizer.zero_grad()

        # Compute output based on samples
        output = model(time_t)

        # Compute data loss (mean squared error)
        loss_data = torch.mean((output - datay_t) ** 2)
        
        # First derivative temperature dT/dt
        dT = torch.autograd.grad(output[:,0], time_t, torch.ones_like(output[:,0]), create_graph=True)[0]
        # First derivative humidity dh/dt
        dh = torch.autograd.grad(output[:,1], time_t, torch.ones_like(output[:,1]), create_graph=True)[0]
        # Residual of differential equations
        rest = (dT - model.talpha + model.tbeta * output[:,0] + model.tgamma *  time_t )
        resh = (dh - model.halpha + model.hbeta * output[:,1] + model.hgamma *  time_t )
        # Set initial conditions
        T0 = tdatay_t[0]
        h0 = hdatay_t[0]
        # Initial condition loss
        loss_init = torch.mean((output[0,0] - T0)**2) + torch.mean((output[0,1] - h0)**2)

        # Compute physics loss
        loss_phys = torch.mean(rest**2) + torch.mean(resh**2)

        # Compute joint loss
        loss = lambda_data * loss_data + lambda_phys * loss_phys + lambda_init * loss_init

        # Backpropagate joint loss
        loss.backward()

        # Perform optimizers step
        optimizer.step()

        # Print epoch info
        print("Epoch ", epoch + 1)
        print("Loss ", loss.item())
        print("Data Loss ", loss_data.item())
        print("Physics Loss ", loss_phys.item())

print("Final Parameters")
print("T Alpha = \t", model.talpha.item())
print("T Beta = \t", model.tbeta.item())
print("T Gamma = \t", model.tgamma.item())
print("H Alpha = \t", model.halpha.item())
print("H Beta = \t", model.hbeta.item())
print("H Gamma = \t", model.hgamma.item())

# Save model weights
torch.save(model.state_dict(), "./pinn_weights.pt")