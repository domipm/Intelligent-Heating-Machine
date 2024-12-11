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
    Future work: Sliding-Windows? So that we can predict (and update prediction) based on incoming data.

'''

# Import all necessary libraries

import  os
import  random
import  datetime

import  numpy               as      np
import  pandas              as      pd
import  matplotlib.pyplot   as      plt
from    matplotlib.lines    import  Line2D

import  scipy.signal        as      signal

from    scipy               import  interpolate

import  torch
import  torch.nn            as      nn

# Set seed for random numbers
seed = datetime.datetime.now().timestamp()
torch.manual_seed(seed)
random.seed(seed)

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
    #x /= np.max(x)
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

# Concatenate temperature and humidity tensor data samples
datay_sample_t = torch.cat((tdatay_sample_t, hdatay_sample_t), dim=1)

# Plot sample data points
plt.plot(tdatax_sample, tdatay_sample, color="tab:blue", markersize=8, linestyle="", marker=".")
plt.plot(hdatax_sample, hdatay_sample, color="tab:orange", markersize=8, linestyle="", marker=".")

# Plot time series
plt.title("Drying Process Evolution")
plt.xlabel(r"Time $[\text{min}]$")
plt.ylabel("Sensor value")
plt.plot(tdatax, tdatay, label = "Temperature", color="tab:blue")
plt.plot(hdatax, hdatay, label = "Humidity", color="tab:orange")

# Define fully-connected neural network
class NN(nn.Module):
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

# Initialize model
model = NN()

# Optimizer hyperparameters
learning_rate = 0.001
# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training hyperparameters
epochs = 10000

# Set model to train
model.train()

# Training loop
for epoch in range(epochs):

    # Reset gradients to zero
    optimizer.zero_grad()

    # Compute output from model using data samples
    out_data = model(tdatax_sample_t)

    # Compute data loss (mean squared error)
    loss = torch.mean((out_data - datay_sample_t) ** 2 )

    # Backpropagate joint loss
    loss.backward()

    # Perform optimizers step
    optimizer.step()

    # Print epoch info
    print("Epoch ", epoch + 1, "Loss ", loss.item())

# Evaluate model over the rest of the domain
model.eval()

# Create output for entire domain
with torch.no_grad():

    # Generate the predictions
    out_data = model(tdatax_t)

out_temp = out_data[:,0].detach().numpy()
out_humi = out_data[:,1].detach().numpy()

plt.plot(tdatax_t.detach().numpy(), out_temp, color="mediumblue", linestyle="-.")
plt.plot(hdatax_t.detach().numpy(), out_humi, color="orangered", linestyle="-.")

# Automatically generated handles
handles, labels = plt.gca().get_legend_handles_labels()
# Create Simple NN line for legend
nn_line = Line2D([0], [0], color="grey", lw=1.5, ls="-.", label="Simple NN")
# Append new handle
handles.extend( [ nn_line ] )
labels.extend( ["Simple NN"])

plt.legend(handles, labels)
plt.show()