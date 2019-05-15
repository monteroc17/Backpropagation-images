import numpy as np
import glob
from pathlib import Path
import os

PATH_TEST = os.path.dirname(os.path.abspath(__file__)) + "/test.txt"

testimage = []
with open(PATH_TEST, "r") as f:
    for line in f:
        for ch in line:
            if (ch == '1') or (ch == '0'):
                testimage.append(int(ch))

class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = len(testimage)
    self.outputSize = 1
    self.hiddenSize = 3

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


  
