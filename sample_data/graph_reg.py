import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.optimize

# Directory where data file stored
fdir = "./data/"

# Directories where data is
fdir = "../sample_data/database/"
files = np.sort(os.listdir(fdir))

# Colors for plots
colors = ["tab:blue", "tab:orange", "tab:green"]

# Empty numpy array to store humidity values
Hdatax = []
Hdatay = []

# Open .csv file as pandas dataframe
df = pd.read_csv(fdir + files[0])
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
        Hdatax.append( getattr(row, "H_tick") )
        Hdatay.append( getattr(row, "H_val") )

# Empty numpy array to store humidity values
THdatax = []
THdatay = []

# Find tick values when drying starts and ends
for row in df.itertuples(index=False, name="Pandas"):
    if getattr(row, "Phase_val") == "Drying":
        drying_start = getattr(row, "Phase_tick")
    if getattr(row, "Phase_val") == "Unloading":
        drying_end = getattr(row, "Phase_tick")
# Find values of temperature (temperature humidity)
for row in df.itertuples(index=False, name="Pandas"):
    row_htick = getattr(row, "TH_tick")
    if row_htick > drying_start and row_htick < drying_end:
        THdatax.append( getattr(row, "TH_tick") )
        THdatay.append( getattr(row, "TH_val") )

plt.plot(THdatax, THdatay)
plt.plot(Hdatax, Hdatay)
#plt.show()

# Polynomial fitting
p_coeff = np.polyfit(THdatax, THdatay, deg=25)
poly = np.poly1d(p_coeff)

x = np.arange(min(THdatax), max(THdatax))
y = poly(x)
#plt.plot(x, y, color="blue", linestyle="--")

# Exponential function
def f(x, A, B, C, D):

    return A*np.exp(-B*x+C)+D

# Fit humidity to exponential
x = np.arange(min(Hdatax), max(Hdatax))
popt, pcov = scipy.optimize.curve_fit(f, Hdatax, Hdatay)
y = f(x,*popt)
plt.plot(x, y, color="orange", linestyle="--")

plt.show()

exit()

for file in os.listdir(fdir):
    fnames.append(file)

# Sort filenames list
fnames = np.sort(fnames)

for fname in fnames:

    f = open(fdir + fname)
    data = json.load(f)

    properties = ["Temperature Humidity", "Humidity"]

    for k, property in enumerate(properties):

        prop = []
        tick = []

        for line in data['Logs']:
            if line['name'] == property: # and line['tick'] > 834219:
                prop.append(line['value'])
                tick.append(line['tick']/1000/60)

        plt.plot(np.asarray(tick, float), np.asarray(prop, float), linestyle="-", linewidth=1, marker=".", markersize=2, label=str(property), alpha=0.25, color=colors[k])

        # Only consider after drying has begun

        plt.plot(np.asarray(tick, float), np.asarray(y_reg, float), linestyle="-", marker='', label=property + " fit", color=colors[k])

    # Plot relevant phases
    tick_phase = []
    phase_type = []

    for line in data['Logs']:
        if line['name'] == "Phase type":
            tick_phase.append(line['tick']/1000/60)
            phase_type.append(line['value'])

    plt.vlines(tick_phase, -10, 120, colors="black", linestyle="--")
    for i in range(len(phase_type)):
        plt.text(x=tick_phase[i]+0.05, y=-10+5*i, s=phase_type[i])

    # Show plot
    plt.title("Temperature and humidity over time (Regression)")
    plt.xlabel(r"Time $t$ [min]")
    plt.ylabel("Value")
    plt.legend(bbox_to_anchor=(1.015, 1.015))
    plt.savefig(str( "./graphs/regression/" + fname.split(".")[0] + ".png" ), dpi=300, bbox_inches="tight")
    plt.close()