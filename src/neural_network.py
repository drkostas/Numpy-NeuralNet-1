from src import FullyConnectedLayer
import numpy as np
from typing import *


class NeuralNetwork:
    def __init__(self, num_layers: int, num_neurons: List[int], activations: List[str],
                 num_inputs: int, loss_function: str, learning_rate: float, weights: List[np.ndarray] = None):
        """
        Initializes a neural network with the given parameters.
        :param num_layers: The number of layers in the network.
        :param num_neurons: A list containing the number of neurons in each layer.
        :param activations: A list containing the activation functions for each layer.
        :param num_inputs: The number of inputs to the network.
        :param loss_function: The loss function to use.
        :param learning_rate: The learning rate to use.
        :param weights: A list containing the weights for each layer.
        """
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.activations = activations
        self.num_inputs = num_inputs
        self.loss_function = loss_function
        self.learning_rate = learning_rate

        if weights is None:
            weights = [(np.random.randn(num_neurons[i], (num_inputs+1) if i == 0 else (num_neurons[i - 1])+1))
                            for i in range(0, num_layers)]

        self.layers = []
        for i in range(num_layers):
            self.layers.append(FullyConnectedLayer(num_neurons[i], self.activations[i],
                                                   num_inputs if i == 0 else num_neurons[i - 1],
                                                   self.learning_rate, weights[i]))

    def calculate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Calculates the output of the network for the given inputs.
        :param inputs: The inputs to the network.
        :return: The output of the network.
        """
        for layer in self.layers:
            inputs = layer.calculate(inputs)
        outputs = np.array(inputs)
        return outputs

    def calculate_loss(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculates the loss of the network for the given inputs and targets.
        :param inputs: The inputs to the network.
        :param targets: The targets for the inputs.
        :return: The loss of the network.
        """
        outputs = np.array(self.calculate(inputs))
        if self.loss_function == "cross_entropy":
            return self.cross_entropy_loss(outputs, targets)
        elif self.loss_function == "mean_squared":
            return self.mean_squared_loss(outputs, targets)
        else:
            raise ValueError("Invalid loss function.")

    @staticmethod
    def cross_entropy_loss(outputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculates the cross entropy loss of the network for the given outputs and targets.
        :param outputs: The outputs of the network.
        :param targets: The targets for the outputs.
        :return: The cross entropy loss of the network.
        """
        return -np.sum(targets * np.log(outputs)) / outputs.shape[0]

    @staticmethod
    def mean_squared_loss(outputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculates the mean squared loss of the network for the given outputs and targets.
        :param outputs: The outputs of the network.
        :param targets: The targets for the outputs.
        :return: The mean squared loss of the network.
        """
        return np.sum((outputs - targets)**2) / outputs.shape[0]

    def loss_derivative(self, outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the loss of the network for the given outputs and targets.
        :param outputs: The outputs of the network.
        :param targets: The targets for the outputs.
        :return: The derivative of the loss of the network.
        """
        if self.loss_function == "cross_entropy":
            return self.cross_entropy_loss_derivative(outputs, targets)
        elif self.loss_function == "mean_squared":
            return self.mean_squared_loss_derivative(outputs, targets)
        else:
            raise ValueError("Invalid loss function.")

    @staticmethod
    def cross_entropy_loss_derivative(outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the cross entropy loss of the network
        for the given outputs and targets.
        :param outputs: The outputs of the network.
        :param targets: The targets for the outputs.
        :return: The derivative of the cross entropy loss of the network.
        """
        return outputs - targets

    @staticmethod
    def mean_squared_loss_derivative(outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the mean squared loss of the network
        for the given outputs and targets.
        :param outputs: The outputs of the network.
        :param targets: The targets for the outputs.
        :return: The derivative of the mean squared loss of the network.
        """
        return 2*((np.array(outputs) - np.array(targets))) / np.array(outputs).shape[0]

    def train(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Trains the network for the given inputs and targets.
        :param inputs: The inputs to the network.
        :param targets: The targets for the inputs.
        :return: None
        """
        outputs = self.calculate(inputs)
        wdeltas = [self.loss_derivative(outputs, targets)]

        for i in range(len(self.layers) - 1, -1, -1):
            wdeltas = self.layers[i].calculate_wdeltas(wdeltas)
