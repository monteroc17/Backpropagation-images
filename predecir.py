import numpy as np
import glob
from pathlib import Path
import pickle
import os
PATH = os.path.dirname(os.path.abspath(__file__)) + "/binaryfiles"

zero = [
  0, 1, 1, 0,
  1, 0, 0, 1,
  1, 0, 0, 1,
  1, 0, 0, 1,
  0, 1, 1, 0
]

one = [
  0, 0, 1, 0,
  0, 0, 1, 0,
  0, 0, 1, 0,
  0, 0, 1, 0,
  0, 0, 1, 0
]

two = [
  0, 1, 1, 0,
  1, 0, 0, 1,
  0, 0, 1, 0,
  0, 1, 0, 0,
  1, 1, 1, 1
]

three = [
  1, 1, 1, 1,
  0, 0, 0, 1,
  0, 1, 1, 1,
  0, 0, 0, 1,
  1, 1, 1, 1
]


predict = [
  1, 1, 1, 1,
  0, 0, 0, 1,
  0, 1, 1, 1,
  0, 0, 0, 1,
  1, 1, 1, 1
]

X = np.array([zero, one, two, three], dtype=float)
y = np.array(([0], [1], [2], [3]), dtype=float)
xPredicted = np.array((predict), dtype=float)

# scale units
y = y/3 # max test score is 100

class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 20
    self.outputSize = 1
    self.hiddenSize = 20

    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propgate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def predict(self):
    print("Predicted data based on trained weights: ");
    print("Input (scaled): \n" + str(xPredicted));
    print("Actual Output: \n" + str((self.forward(xPredicted))*3));
    #print("Rounded Output: \n" + str(round((self.forward(xPredicted))*3)));

    def save_weights(self, peso_c_oculta, peso_c_salida):
        with open('w1.txt', 'wb') as file:
            pickle.dump(peso_c_oculta, file)
        with open('w2.txt', 'wb') as file:
          pickle.dump(peso_c_salida, file)

    def load_weights(self):
        with open('w1.txt', 'rb') as file:
          self.W1 = pickle.load(file)
        with open('w2.txt', 'rb') as file:
          self.W2 = pickle.load(file)

NN = Neural_Network()
for i in range(10000): # trains the NN 100,000 times
  print("#" + str(i) + "\n")
  print("Input: \n" + str(X))
  print("Actual Output: \n" + str(y*3))
  print("Predicted Output: \n" + str(NN.forward(X)*3))
  print("Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))))# mean sum squared loss
  print("\n")
  NN.train(X, y)

NN.predict()