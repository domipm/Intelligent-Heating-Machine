import json
import numpy as np
import matplotlib.pyplot as plt

fname = "2050.json"

file = open(fname)
data = json.load(file)

properties = ["Temperature Humidity", "Humidity"]
for property in properties:

    prop = []
    tick = []

    for line in data['Logs']:
        if line['name'] == property:
            prop.append(line['value'])
            tick.append(line['tick']/1000/60)

    plt.plot(np.asarray(tick, float), np.asarray(prop, float), linestyle="-", linewidth=1, marker=".", label=str(property))

plt.title("Properties over Time")
plt.xlabel(r"Time $t$ [min]")
plt.ylabel("Value")

plt.legend()

plt.savefig(str( "./gph/" + fname.split(".")[0] + ".png" ), dpi=300)
plt.show()