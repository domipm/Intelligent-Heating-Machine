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
    ├ data_pre.py -- Used for pre-processing measured data  
    ├ README.md 
    ├ DOCS.md
    └ .gitignore

The folder `sample_data` contains a small subset of the total data available in `measurements` and was used mainly in the first development period to test the code, however the structure of both folders is the same. Inside, there all folders that separate the data into `all` files as well as `train` and `test` splits. The scripts `graph*.py` are also mainly used for visualization purposes of the data and how the Savitzky-Golay and Exponential filtering affect it.

The folder `simple_net` was also mainly used for testing new approaches in the development of the neural network, and contains some prototyping code. The main code, however, is located in the `pinn` folder, where the final implementation. The following is a step-by-step description of the workflow for this project.

### Data Pre-Processing

The measured data, located in a directory `m_dir` (which corresponds to the `/data/` folder) in `*.json` format, is read by the `./data_pre.py` script. This sorts the files and for each one of them, creates a `Pandas` dataset that includes the tick times and values for temperature, humidity, temperature humidity, and phase type properties found in the raw `*.json` files. As the last step, these databases are then saved into `*.csv` files in the `d_dir` directory (`/database/` folder).

### PINN Dataset

Next, we go into the `./pinn/` folder that contains the code for the Physics-Informed Neural Network. Here, the script `pinn_dataset.py` defines the custom data loader for our files. The class `DryingDataset`, which inherits from `PyTorch`'s `Dataset` component reads all the pre-processed `.csv` files generated with the previous script and then reads the temperature and humidity values after the drying phase has begun. These arrays are then passed on to the function `data_clean` that does the following:
1. Removes the first few datapoints (as these tend to present some unwanted noise).
2. Removes any duplicated entries of values
3. "Normalizes" the time range, that is, sets time $t = 0$ when drying begins by subtracting the minimum time value.
4. Performs interpolation of the data, using linear splines. This ensures we can sample the same amount of data points for each file, effectively "normalizing" the number of ticks considered.
5. Finally, the Savitzky-Golay filtering is applied to smooth-out the data and remove any spikes which may be due to sensor noise.

Once the data is cleaned up, the time, temperature, and humidity values are converted to `PyTorch` tensor types, and the temperature and humidity values are concatenated. This way, using the `__getitem__()` function on an instance of the class will return a tensor that contains all the time values (always of fixed size, `250`) and the values of temperature and humidity as another tensor with compatible dimensions.

### PINN Model

In the `pinn_model.py` script, the architecture of the neural network is defined as a simple fully-connected network in the `PINN` class, with four linear layers with a hidden size of `64` for the first three and `2` for the last, output layer. Most importantly, here are also defined the learnable parameters that will be used by the physics equations for describing the system. Mainly, the parameters have been called by prefix `t` for temperature and `h` for humidity and `alpha`, `beta`, `gamma`, etc. and all allow for the model to learn them, having `requires_grad=True`.

### PINN Train Loop

`pinn_train.py` contains the necessary code to train the network based on the measured data. Here's how it works. The model first loads the training dataset and creates an instance of the dataset class. From this, an array is created named `dataset` that will allow us to loop over the files and randomly arrange them. After initializing the model and setting it to train mode, using the `Adam` optimizer with a pre-defined `learning_rate` hyperparameter, the training is then done over multiple `epochs`.

Another set of hyperparameters defined as `lambda_x` are the associated weights to each loss term, so we can control how sensitive the model is to optimizing the loss of the physics equations over the data, for example. In total, we have three `lambda` parameters that each control the weight associated to each type of loss: physics loss, data loss, and initial conditions loss.

The data loss `loss_data` is defined simply as the mean squared error between the predicted output of the model and the actual values. The `output` is computed using the `time` array values as input to the model, and letting it generate a two-dimensional tensor array containing the temperature and humidity. 

The physics loss `loss_physics` is computed by first taking the derivative of each variable with time using `PyTorch`'s `autograd.grad()` function. With this, we can write the residuals associated to each differential equation for the evolution of temperature and humidity of the drying machine's exhaust air. The final loss is then the sum of the mean of each residual squared.

Finally, the initial conditions for each equation have been considered to be equal to the first measured datapoints, so that, for temperature, $T(t=0) = T_0$, and for humidity, $H(t=0)=H_0$. We ensure our prediction fall close to this initial conditions by including a loss term that computes the mean squared error between the first values of the inputs and the measured initial values.

The final and total loss function will be a weighted sum of these three previously-defined individual loss functions, with weights manually adjusted using the `lambda` parameters.

Then, we backpropagate the loss and perform a step of the optimizer as usual. Once training has finalized, the weights are saved into a `.pt` file and a plot is created that tracks how the loss has evolved over the training epochs.

### PINN Validation

For the validation of the model, we now consider the test dataset and load the pre-trained weights of the model. 

The testing is first done using a single measurement file. We begin by considering the first few points of the time-series (for example, the first two minutes of drying). These time values are then ran through the model to fine-tune the weights to the specific configuration of the current drying process. The idea is that during training, we have learned the general patterns of how temperature and humidity will behave, but now by fine-tuning the weights over the first few datapoints available, we can learn the more subtle changes in configuration that will allow to improve the predictions. To make sure we don't modify the weights too much, we only perform this fine-tuning for a small amount of epochs. The computation of the loss during this process is exactly the same as during training.

One the model has been fine-tuned to the specific drying period, we set it to evaluation mode and, without computing gradients, generate a prediction using the output of the model over the full time domain of the file (in principle, this time lenght could be arbitrarily long).

Finally, a plot is generated that shows the prediction of the model along with the actual measured data and the datapoints used for fine-tuning.

As an additional step, another loop was included that performs this same exact procedure over all the test files and calculates the average testing loss.