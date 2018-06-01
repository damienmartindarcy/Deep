# Neural network without tensorflow
# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6


# input layer x
# output layer y
# arbitrary number of hidden layers
# Weights and biases between each layer w and b
# Choice of activiation function for each hidden layer
# Training the neural network = finetuning weights and biases from the input data
# Loss function is to evaluate the goodness of the model .. here we use simple sum of squares error
# The point of training is to find the best set of weights and biases that minimise the loss function


# At each iteration ....
# Calculating the predicted output = feedforward
# Updating the weights and biases = backpropagation
        
import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

# We input x1, x2, x3 with four values each as input ... then we get four values of y
# The algorithm is run for 1500 iterations 
# The loss function monotonically decreases towards a minumum        
        
if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork(X,y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()

    print(nn.output) 
    
# Going forward .. we can use other activiation functions besides sigmoid
# We can use a learning rate whilst training the neural network
# Use convolutions for image classification tasks    
       
        
        