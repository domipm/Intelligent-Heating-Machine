# Script to generate graphs that showcase the final, filtered data

import  json
import  os
import  numpy               as np
import  matplotlib.pyplot   as plt
from    scipy               import  interpolate

from scipy.signal import savgol_filter

# Directory where data file stored
fdir = "./data/"
# Directory to save plots
gdir = "./graphs/savitzky-golay"

# List of all files available
fnames = []

# Savitzky-Golay filter parameters
window = 25
order = 3

# Colors for plots
colors = ["tab:blue", "tab:orange", "tab:green"]

for file in os.listdir(fdir):
    fnames.append(file)

# Sort filenames list
fnames = np.sort(fnames)

for fname in fnames:

    f = open(fdir + fname)
    data = json.load(f)

    properties = ["Temperature Wash Tank", "Humidity"]

    for k, property in enumerate(properties):

        prop = []
        tick = []

        for line in data['Logs']:
            if line['name'] == property: # and line['tick'] > 834219:
                prop.append(line['value'])
                tick.append(line['tick']/1000/60)

        # Interpolate data
        interp = interpolate.interp1d(tick, prop, kind="linear")
        x = np.linspace(min(tick), max(tick), 1000)
        y = interp(x)

        # Smooth data and plot smoothed-out curve
        y_filter = savgol_filter(y, window_length=window, polyorder=order)
        plt.plot(x, y_filter, linestyle="-", marker='', label=property.split(" ")[0], color=colors[k])

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
    plt.title("Drying Process (Batch {}) Filtered".format(fname.split(".")[0]))
    plt.xlabel(r"Time $t$ [min]")
    plt.ylabel("Value")
    #plt.legend(bbox_to_anchor=(1.015, 1.015))
    plt.legend()
    plt.savefig(str( "./graphs/filter/" + fname.split(".")[0] + ".pdf" ), bbox_inches="tight")
    plt.close()