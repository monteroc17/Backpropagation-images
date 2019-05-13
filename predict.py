from math import exp
import pickle
import glob

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	if activation < 0:
		return 1.0 - 1.0 / (1.0 + exp(activation))
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row[0]
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Test making predictions with the network
files = glob.glob("D:/Documents/Semestres/7/IA/Examen1/Backpropagation-images/binaryfiles/*.txt")
dataset = []
filedata = []

for myFile in files:
    with open(myFile, "r") as f:
        for line in f:
            for ch in line:
                if (ch == '1') or (ch == '0'):
                    filedata.append(float(ch))
    dataset.append([filedata, 1])


with open('weights.txt', 'rb') as file:
    network = pickle.load(file)
# network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
# 	[{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]
for row in dataset:
	prediction = predict(network, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))