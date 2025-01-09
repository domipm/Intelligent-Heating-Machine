import  json
import  os
import  numpy   as np
import  pandas  as pd

# Directory where measurements are
m_dir = "./measurements/data/"
# Directory where dataframes are to be saved
d_dir = "./measurements/database/"

# Sorted array of all file names available in directory
files = np.sort(os.listdir(m_dir))

# For each file, create pandas dataset with following columns:
# T_tick, TempWT, H_tick, Hum, TH_tick, THum (other parameters in future maybe)

# Go over all files and create relevant .csv databases
for k, file in enumerate(files):

    # Read first file
    f = open(m_dir + file)
    data = json.load(f)

    # Create relevant arrays (X_tick time of data, X_val value)
    T_tick = []
    T_val = []
    H_tick = []
    H_val = []
    TH_tick = []
    TH_val = []
    Phase_tick = []
    Phase_val = []

    # Read all the relevant data
    for line in data['Logs']:
        # Save washing/drying phases and when they happen
        if (line['name'] == "Phase type"):
            # Tick when phase changes (Loading, Prewash, Disinfection, Drying, Unloading)
            Phase_tick.append(line['tick'])     # Tick value
            Phase_val.append(line['value'])     # Property value (phase name)
        # Temperature of wash tank
        if (line['name'] == "Temperature Wash Tank"):
            T_tick.append(float(line['tick'])) # Tick value
            T_val.append(float(line['value'])) # Property value
        # Humidity of air coming out of chamber
        if (line['name'] == "Humidity"):
            H_tick.append(float(line['tick']))  # Tick value
            H_val.append(float(line['value']))  # Property value
        # Temperature of air coming out of chamber
        if (line['name'] == "Temperature Humidity"):
            TH_tick.append(float(line['tick'])) # Tick value
            TH_val.append(float(line['value'])) # Property value

    # Dictionary of lists
    values = {'T_tick': T_tick, 'T_val': T_val, "H_tick": H_tick, "H_val": H_val, "TH_tick": TH_tick, "TH_val": TH_val, "Phase_tick": Phase_tick, "Phase_val": Phase_val}
    # Convert dictionary of lists into pandas dataframe (fill unequal lengths with NaN values)
    dataframe = pd.DataFrame({ key:pd.Series(value) for key, value in values.items() })
    
    # Save dataframe to file
    dataframe.to_csv(d_dir + str(file.split(".")[0]) + ".csv", index=False)