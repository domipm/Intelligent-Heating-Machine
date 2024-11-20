import json
import os
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Directory where data file stored
fdir = "./data/"
# Directory to save plots
gdir = "./graphs/exp-smooth/"

# List of all files available
fnames = []

# Colors for plots
colors = ["tab:blue", "tab:blue", "tab:orange", "tab:orange", "tab:green", "tab:green"]

for file in os.listdir(fdir):
    fnames.append(file)

# Exponential smoothing alpha value
alpha = 0.75

# Sort filenames list
fnames = np.sort(fnames)

for fname in fnames:

    f = open(fdir + fname)
    data = json.load(f)

    properties = ["Temperature Humidity", "Humidity", "Temperature Wash Tank"]

    for k, property in enumerate(properties):

        prop = []
        tick = []

        for line in data['Logs']:
            if line['name'] == property: # and line['tick'] > 834219:
                prop.append(line['value'])
                tick.append(line['tick']/1000/60)

        # Plot original values
        plt.plot(np.asarray(tick, float), np.asarray(prop, float), linestyle="-", linewidth=1, marker=".", markersize=2, label=str(property), alpha=0.25, color=colors[k])
        # Convert values to float
        prop = np.array(prop, dtype=float)

        # Fit model to data
        ses_model = SimpleExpSmoothing(prop).fit(smoothing_level = alpha, optimized = False)
        plt.plot(tick, ses_model.fittedvalues, label=property + " fit", color=colors[k])

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
    plt.title("Temperature and humidity over time (Exponential smoothing)")
    plt.xlabel(r"Time $t$ [min]")
    plt.ylabel("Value")
    plt.legend(bbox_to_anchor=(1.015, 1.015))
    plt.savefig(str( gdir + fname.split(".")[0] + ".png" ), dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()