import json
import numpy as np
import matplotlib.pyplot as plt

fname = "2050.json"

file = open(fname)
data = json.load(file)

# Plot relevant properties
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

plt.savefig(str( "./gph/" + fname.split(".")[0] + ".png" ), dpi=300)
plt.show()