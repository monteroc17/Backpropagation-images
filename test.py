import numpy as np
import glob
from pathlib import Path
import pickle

PATH = "D:/Documents/Semestres/7/IA/Examen1/Backpropagation-images/binaryfiles"
PATH_TEST = "D:/Documents/Semestres/7/IA/Examen1/Backpropagation-images/test.txt"
# G:/Files/TEC/IA/Backpropagation-images/binaryfiles                       <-- Daniel
# D:/Documents/Semestres/7/IA/Examen1/Backpropagation-images/binaryfiles   <--Josue

# image path
files = glob.glob(PATH + "/*.txt")

inputs = []
res_esperados = []

for myFile in files:
    imagedata = []
    print(myFile)
    with open(myFile, "r") as f:
        for line in f:
            for ch in line:
                if (ch == '1') or (ch == '0'):
                    imagedata.append(int(ch))
        # f.close()
    inputs.append(imagedata)
    imagedata = []

X = np.array(inputs, dtype=int)


y = np.array([[0], [1], [2]], dtype=int)

# scale units
# X = X/np.amax(X, axis=0) # maximum of X array
# y = y/100 # max test score is 100


class Neural_Network(object):
    def __init__(self):
        # parameters
        self.inputSize = len(inputs[0])
        self.outputSize = 1
        self.hiddenSize = 4

        # weights
        # (3x2) weight matrix from input to hidden layer
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        # (3x1) weight matrix from hidden to output layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self, X):
        # forward propagation through our network
        # dot product of X (input) and first set of 3x2 weights
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)  # activation function
        # dot product of hidden layer (z2) and second set of 3x1 weights
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)  # final activation function
        return o

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        # backward propgate through the network
        self.o_error = y - o  # error in output
        # applying derivative of sigmoid to error
        self.o_delta = self.o_error*self.sigmoidPrime(o)

        # z2 error: how much our hidden layer weights contributed to output error
        self.z2_error = self.o_delta.dot(self.W2.T)
        # applying derivative of sigmoid to z2 error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)

        # adjusting first set (input --> hidden) weights
        self.W1 += X.T.dot(self.z2_delta)
        # adjusting second set (hidden --> output) weights
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

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
#NN.load_weights()
for i in range(20000):  # trains the NN 1,000 times
    print("Input: \n" + str(X))
    print("Resultados Esperados: \n" + str(y))
    print("Resultados obtenidos: \n" + str(NN.forward(X)))
    # mean sum squared loss
    print("Error: \n" + str(np.mean(np.square(y - NN.forward(X)))))
    print("\n")
    NN.train(X, y)
NN.save_weights(NN.W1, NN.W2)
image = []
with open(PATH_TEST, "r") as f:
    for line in f:
        for ch in line:
            if (ch == '1') or (ch == '0'):
                image.append(int(ch))

test = np.array([image], dtype=int)
print("--------------------TEST---------------------")
print("Resultados Esperados: \n" + str(0))
print("Resultados obtenidos: \n" + str(NN.forward(test)))
