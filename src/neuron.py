import numpy

class Neuron:
    # ACT_FUNCTION, NUM_INPUTS, LEARNING_RATE, [INIT_WEIGHTS]

    def __init__(*args, **kwargs):
        # Initializes all input vars
        args[0].act_funct = args[1]
        args[0].num_inputs = args[2]
        args[0].learning_rate = args[3]
        if len(args) > 4:
            args[0].weights = args[4]
        else:
            # Calculate random weights if no initial weights are given.
            args[0].weights = (numpy.random.rand(args[0].num_inputs)- .5)*2

    # Uses the saved net value and activation function to return the output of the node
    def activate(self):
        if self.act_funct == "linear":
            self.output = self.net
        elif self.act_funct == "sigmoid":
            self.output = 1/(1+numpy.exp(-self.net))
        return self.output

    # Receives a vector of inputs and determines the nodes output using
    # the stored weights and the activation function
    def calculate(self, inputs):
        self.inputs = inputs
        self.net = numpy.sum(inputs*self.weights)
        return self.activate()

    # Returns the derivative of the activation function using the previously calculated output.
    def activation_derivative(self):
        if self.act_funct == "linear":
            return 1;
        elif self.act_funct == "sigmoid":
            return self.output*(1-self.output)

    # Calculates and saves the partial derivative with respect to the weights
    def derivative(self, delta):
        self.partial_der = self.weights*delta

    # Calculates the new delta*w and calls upon the derivative function
    def calcpartial_derivative(self,deltaw_1):
        delta = numpy.sum(deltaw_1)*self.activation_derivative()
        self.derivative(delta)
        return delta*self.weights

    # Updates the nodes weights using the saved partial derivatives and learning rate.
    def update_weights(self):
        self.weights = self.weights-self.learning_rate*self.partial_der
