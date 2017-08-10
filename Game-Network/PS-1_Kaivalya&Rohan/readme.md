
# Neural Networks for GamePlaying AIs

A detailed report about this project has been made and submitted. This file contains only basic operating instructions.

## Running the Flask server
This is the simplest use of this project. It uses a pretrained model on pregenerated data to demonstrate the neural network in action.

1. Open the terminal and execute `python3 neuralnetworks.py` to start the flask server.
2. In the browser, navigate to `127.0.0.1:5000` 


## Using the Game Console.

The main purpose of this module is to generate training and testing data

Open the terminal and execute `python3 console.py`. This creates 2 files: `train.csv` and `test.csv`. 



## Training the neural Network

By default, the `neuralnetworks.py` file runs the flask server. By uncommenting the `main()` statement and commenting the `app.run()`, this module can be used to train the network instead

To train the neural network, all parameters can be passed directly from the command line except for the network shape and activation function. The current default for these is a single hidden layer of 5 neurons, and a modified sigmoid activation function. Other parameters can be accessed via the command line as follows:

* `training-data-size` specifies the no. of data points to train the model with
* `batch-size` specifies the batch size for each iteration
* `epochs` specifies the no. of times the whole data set is fed to the network
* `alpha` specifies the learning rate for gradient descent
