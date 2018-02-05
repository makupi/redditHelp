import numpy as np


def activation(x, func="gwavelet", deriv=False):
    z = np.zeros(x.shape)
    if func == "sigmoid":
        z = 1.0 / (1 + np.exp(-x))
        if deriv:
            z = z * (1 - z)

    if func == "gwavelet":
        z = x * np.exp(-x*x/2)
        if deriv:
            z = z * (1 - z*z)

    if func == "tanh":
        z = np.tanh(x)
        if deriv:
            z = 1 - z * z

    if func == 'xtanh':
        z = 20*np.tanh(0.1*x)
        if deriv:
            z = 2/(np.cosh(0.1*x) * np.cosh(0.1*x))

    if func == "res":
        z = 1.0 / (1 + np.exp(-x*x*x))
        if deriv:
            z = 3 * z * x * x * (1 - z)

    if func == "leaky_ReLU":
        z = np.zeros(x.shape)
        z = np.maximum(0.01*x, x)

        if deriv:
            z[z >= 0] = 1
            z[z < 0] = 0.01

    return z


class Layer:

    def __init__(self, height, inputs, activation="tanh"):
        self.x = np.zeros((1, height))
        self.height = height
        self.inputs = inputs
        # np.random.seed(69)
        self.weights = (np.random.random((self.height, self.inputs)))
        self.deriv = np.zeros(self.weights.shape)
        self.activation = activation

    #One forward movement through the network
    def propagate(self, inputs):
        self.x = activation(np.dot(self.weights, inputs), func=self.activation)
        return self.x


class NeuralNetwork:

    def __init__(self, inputs=1, outputs=1, hidden_layers=5, hidden_layer_height=10, activation="gwavelet", layers=None):
        self.activation = activation
        self.hidden_layer_height = hidden_layer_height #These two variables are pointless if you include the layers variable at the end
        self.hidden_layers = hidden_layers
        self.hidden_layers -= 1
        if layers==None:
            self.layers = [Layer(hidden_layer_height, inputs, activation=activation)]
            layer_count = 2
            while layer_count <= hidden_layers:
                layer = Layer(hidden_layer_height, hidden_layer_height, activation=activation)
                self.layers.append(layer)
                layer_count = layer_count + 1

            self.layers.append(Layer(outputs, hidden_layer_height, activation=activation))
        else:
            self.layers=[Layer(layers[0], inputs, activation=activation)]
            for i in range(len(layers) - 1):
                layer = Layer(layers[i+1], layers[i], activation=activation)
                self.layers.append(layer)
            self.layers.append(Layer(outputs, layers[-1], activation=activation))

    #Full forward propagation through the Network
    def forward(self, input):
        for layer in self.layers:
            i = self.layers.index(layer)
            if i == 0:
                layer_output = layer.propagate(input)
            else:
                if i == len(self.layers) - 1:
                    return layer.propagate(layer_output)
                else:
                    layer_output = layer.propagate(layer_output)


    #back propagation
    def train(self, target, training_vector, alpha, iterations=1):
        iteration_count = 0

        while iteration_count < iterations:
            output = self.forward(training_vector)
            error = np.average(np.abs(output - target))

            # print("%s -> %s T %s" %(training_vector,str(output),str(target)))
            # print("%s" % (str(error)))

            self.forward(training_vector)
            for layer in reversed(self.layers):
                i = self.layers.index(layer)

                if i == len(self.layers) - 1:
                    E = (layer.x - target)
                    di = (E * activation(np.dot(layer.weights, self.layers[i - 1].x), deriv=True))
                    deriv = np.dot(di, np.transpose(self.layers[i - 1].x))

                else:
                    if i == 0:
                        di = (np.dot(np.transpose(self.layers[i+1].weights), di) * activation(np.dot(layer.weights, training_vector), deriv=True))
                        deriv = np.dot(di, np.transpose(training_vector))

                    else:
                        di = (np.dot(np.transpose(self.layers[i+1].weights), di) * activation(np.dot(layer.weights, self.layers[i - 1].x), deriv=True))
                        deriv = np.dot(di, np.transpose(self.layers[i - 1].x))

                layer.weights -= alpha * deriv

            iteration_count = iteration_count + 1
        return output


##USE CASE

#XOR problem
input = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
target = np.array([[0, 1, 1, 0]])

#The middle 2 parameter are pointless if you include the final parameter. As they both define the Network parameters so [5, 5, 5] means it is a network with 3 hidden layers all of size 5
n1 = NeuralNetwork(2, 1, 3, 4, activation='tanh', layers=[5, 5, 5])
output = n1.train(target, input, 0.3, 10000)
print(output)
exit()