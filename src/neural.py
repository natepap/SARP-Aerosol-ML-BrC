import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import model_selection

rootDir = 'C:/Stuff/Projects/SARP-Aerosol-ML-BrC/Data/'
rawPath = rootDir + 'Raw/SAGAMERGE/'
cleanPath = rootDir + 'Cleaned/'
processPath = rootDir + 'Processed/'
featurePath = rootDir + 'Features/'

'''
This file is for all neural network attempts - from scratch and from libraries

Includes a from scratch class for constructing a neural network and also allows us to define the 
activation function
'''

#TODO: Normalize inputs/outputs - values are too small for weights to update in backprop()


# Load dataset (currently 429 input features)
input_data = pd.read_csv(processPath + 'input').to_numpy()
output_data = pd.read_csv(processPath + 'output').to_numpy()

x_train, x_test, y_train, y_test = model_selection.train_test_split(input_data, output_data, train_size=0.7)

'''
# sigmoid activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))


def sigmoid_derivative(p):
    return p * (1 - p)


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.output

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.feedforward()
        self.backprop()



NN = NeuralNetwork(x, y)

for i in range(1500):  # trains the NN 1,000 times
    if i % 100 == 0:
        print("for iteration # " + str(i) + "\n")
        print("Input : \n" + str(x))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(NN.feedforward()))
        print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))  # mean sum squared loss
        print("Weights: \n" + str(NN.weights1) + str(NN.weights2))
        print("\n")

    NN.train(x, y)
'''

inputs = keras.Input(shape=429,)

dense = layers.Dense(64, activation='relu')
x = dense(inputs)

x = layers.Dense(64, activation='relu')(x)

outputs = layers.Dense(1)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='brcmodel')

model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.RMSprop(),
              metrics=keras.metrics.MeanAbsolutePercentageError(),)

history = model.fit(x_train, y_train, epochs=2000, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
