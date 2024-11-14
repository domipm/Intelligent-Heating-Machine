import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Directory where data file stored
fdir = "./data/"
# Directory to save plots
gdir = "./graphs/"

# List of all files available
fnames = []

for file in os.listdir(fdir):
    fnames.append(file)

# Sort filenames list
fnames = np.sort(fnames)

for fname in fnames:

    plt.close()

    f = open(fdir + fname)
    data = json.load(f)

    properties = ["Temperature Humidity", "Humidity", "Temperature Wash Tank"]

    for property in properties:

        prop = []
        tick = []

        for line in data['Logs']:
            if line['name'] == property: # and line['tick'] > 834219:
                prop.append(line['value'])
                tick.append(line['tick']/1000/60)

        plt.plot(np.asarray(tick, float), np.asarray(prop, float), linestyle="-", linewidth=1, marker=".", label=str(property))

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
    plt.title("Properties over Time")
    plt.xlabel(r"Time $t$ [min]")
    plt.ylabel("Value")

    plt.legend()

    plt.savefig(str( gdir + fname.split(".")[0] + ".png" ), dpi=300)