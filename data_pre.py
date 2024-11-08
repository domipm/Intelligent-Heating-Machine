# Script used to pre-process the data from measurements

import numpy
import json
import pandas as pd
import matplotlib.pyplot as plt

# Directory where measurements are
m_dir = "./measurements/"
# Directory where to save .csv files (time series for each measurement, pre-processed)
d_dir = "./database/"

# 1. Load relevant parameters into program (focus on temperature, humidity, time)
# 2. Normalize time series and data values (if needed?)
# 3. Save output .csv file to data directory