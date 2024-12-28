# Code Documentation

This file is to be used as documentation for the code base developed for this project. The aim is to describe in detail what each script does and how each of its components work. 

The file structure of this repository is as follows:

    ./Intelligent-Heating-Machine/
    ├ diagram/      -- Diagram of drying machine
    ├ measurements/ -- Excluded from GitHub, includes all measurement data
      ├ data/           -- Raw data (*.json)
      ├ database/       -- Pre-processed data (*.csv)
      └ graphs/         -- Visualization graphs (*.png)
    ├ sample_data/  -- Small sample of data. Scripts used for visualization purposes.
        ├ data/              -- Raw data (*.json)
        ├ database/          -- Pre-processed data (*.csv)
        ├ graphs/            -- Visualization graphs (*.png)
        ├ graph_expsmooth.py -- Applies Exponential Smoothing filter to data and generates graph
        ├ graph_sg.py        -- Applies Savitzky-Golay filter to data and generates graph
        ├ graph.py           -- Generates graphs of all data (for filtering purposes)
    ├ pinn/  -- Code used for Physics-Informed Neural Network (PINN)
        ├ output/   -- Output graphs (*.png) and model weights (*.pt)
        ├ pinn_dataset.py -- Dataset class for (*.csv) files to feed the network
        ├ pinn_model.py -- Network model class that defines the architecture and learnable parameters
        ├ pinn_train.py -- Training script for model
        └ pinn_test.py -- Testing script for verifying model
    ├ simple_net/ -- Used for comparison with standard neural network model (Some testing scripts)
    ─ data_pre.py -- Used for pre-processing measured data
    - .gitignore  -- GitHub ignore files
    - README.md   -- This file

Let's begin with how the workflow looks like. First, the measured data, located in a directory `m_dir` (which corresponds to the `/data/` folder) in `*.json` format, is read by the `./data_pre.py` script. This sorts the files and for each one of them, creates a `Pandas` dataset that includes the tick times and values for temperature, humidity, temperature humidity, and phase type properties found in the raw `*.json` files. As the last step, these databases are then saved into `*.csv` files in the `d_dir` directory (`/database/` folder).

Next, we go into the `./pinn/` folder that contains the code for the Physics-Informed Neural Network. Here, the script `pinn_dataset.py` defines the custom data loader for our files. The class `DryingDataset`, which inherits from `PyTorch`'s `Dataset` component reads all the pre-processed `.csv` files generated with the previous script and then reads the temperature and humidity values after the drying phase has begun. These arrays are then passed on to the function `data_clean` that does the following:
1. Removes the first few datapoints (as these tend to present some unwanted noise).
2. Removes any duplicated entries of values
3. "Normalizes" the time range, that is, sets time $t = 0$ when drying begins by subtracting the minimum time value.
4. Performs interpolation of the data, using linear splines. This ensures we can sample the same amount of data points for each file, effectively "normalizing" the number of ticks considered.
5. Finally, the Savitzky-Golay filtering is applied to smooth-out the data and remove any spikes which may be due to sensor noise.

Once the data is cleaned up, the time, temperature, and humidity values are converted to `PyTorch` tensor types, and the temperature and humidity values are concatenated. This way, using the `__getitem__()` function on an instance of the class will return a tensor that contains all the time values (always of fixed size, `250`) and the values of temperature and humidity as another tensor with compatible dimensions.

In the `pinn_model.py` script, the architecture of the neural network is defined as a simple fully-connected network in the `PINN` class, with four linear layers with a hidden size of `64` for the first three and `2` for the last, output layer. Most importantly, here are also defined the learnable parameters that will be used by the physics equations for describing the system. Mainly, the parameters have been called by prefix `t` for temperature and `h` for humidity and `alpha`, `beta`, `gamma`, etc. and all allow for the model to learn them, having `requires_grad=True`.