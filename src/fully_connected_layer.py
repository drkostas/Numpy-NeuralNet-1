from src import Neuron
import numpy as np
from typing import *


class FullyConnectedLayer:
    def __init__(self, num_neurons: int, activation: str, num_inputs: int, lr: float,
                 weights: np.ndarray = None):
        """
        Initializes a fully connected layer.
        :param num_neurons: Number of neurons in the layer
        :param activation: Activation function
        :param num_inputs: Number of inputs to each neuron
        :param lr: Learning rate
        :param weights: Weights of the layer
        """
        self.num_neurons = num_neurons
        self.activation = activation
        self.num_inputs = num_inputs
        if weights is None:
            self.weights = np.random.rand(num_neurons, num_inputs)
        else:
            self.weights = weights
        self.neurons = []
        for neuron_ind in range(self.num_neurons):
            self.neurons.append(Neuron(self.activation, self.num_inputs, lr, weights[neuron_ind]))
        self.lr = lr

    def calculate(self, inputs: np.ndarray) -> List:
        """
        Calculates the output of the layer.
        :param inputs: Inputs to the layer
        :return: Output of the layer
        """
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate(inputs))
        return outputs

    def calculate_wdeltas(self, wdeltas_next: List) -> List:
        """
        Calculates the weight deltas of the layer.
        :param wdeltas_next: Weight deltas of the next layer
        :return: Weight deltas of the layer
        """
        wdeltas = []
        for ind, neuron in enumerate(self.neurons):
            wdelta = neuron.calcpartialderivative(wdeltas_next[ind])
            neuron.update_weights()
            wdeltas.append(wdelta)
        return wdeltas
