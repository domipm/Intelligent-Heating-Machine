import  os
import  torch

import  numpy               as      np
import  pandas              as      pd
import  scipy.signal        as      signal
from    scipy               import  interpolate

from    torch.utils.data    import  Dataset

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