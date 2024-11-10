# Script used for all pre-processing of the data

'''

TO-DO / IDEAS:

    - Include current washing/drying phase for RNN to use
    - Include time of drying ("program duration" in data files)
    - Smooth graph out (interpolation/spline) to have continuous time steps
      (equal time ticks for all parameters)
    
'''

import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory where measurements are
m_dir = "./measurements/data/"
# Directory where some sample data can be found
s_dir = "./data_sample/data/"
# Directory where dataframes are to be saved
d_dir = "./database/"

# Sorted array of all file names available in directory
files = np.sort(os.listdir(s_dir))

# For each file, create pandas dataset with following columns:
# T_tick, TempWT, H_tick, Hum, TH_tick, THum (other parameters in future maybe)

# Go over all files and create relevant .csv databases
for k, file in enumerate(files):

    # Read first file
    f = open(s_dir + file)
    data = json.load(f)

    # Create relevant arrays
    T_tick = []
    T_val = []
    H_tick = []
    H_val = []
    TH_tick = []
    TH_val = []

    # Read all the relevant data
    for line in data['Logs']:

        if (line['name'] == "Temperature Wash Tank"):

            T_tick.append(float(line['tick']))
            T_val.append(float(line['value']))

        if (line['name'] == "Humidity"):

            H_tick.append(float(line['tick']))
            H_val.append(float(line['value']))

        if (line['name'] == "Temperature Humidity"):

            TH_tick.append(float(line['tick']))
            TH_val.append(float(line['value']))

    # Dictionary of lists
    values = {'T_tick': T_tick, 'T_val': T_val, "H_tick": H_tick, "H_val": H_val, "TH_tick": TH_tick, "TH_val": TH_val}
    # Convert dictionary of lists into pandas dataframe (fill unequal lengths with NaN values)
    dataframe = pd.DataFrame({ key:pd.Series(value) for key, value in values.items() })
    
    # Save dataframe to file
    dataframe.to_csv(d_dir + str(file.split(".")[0]) + ".csv", index=False)