import traceback
import argparse

import numpy as np
from src import Configuration, ColorLogger, Neuron, FullyConnectedLayer, NeuralNetwork
from typing import *

logger = ColorLogger(logger_name='Main', color='yellow')


def get_args() -> argparse.Namespace:
    """Setup the argument parser

    Returns:
        argparse.Namespace:
    """
    parser = argparse.ArgumentParser(
        description='Project 1 for the Deep Learning class (COSC 525). '
                    'Involves the development of a FeedForward Neural Network.',
        add_help=False)
    # Required Args
    required_args = parser.add_argument_group('Required Arguments')
    config_file_params = {
        'type': argparse.FileType('r'),
        'required': True,
        'help': "The configuration yml file"
    }
    required_args.add_argument('-d', '--dataset', required=True,
                               help="The datasets to train the network on. "
                                    "Options (defined in yml): [and, xor, class_example]")
    required_args.add_argument('-n', '--network', required=True,
                               help="The network configuration to use. "
                                    "Options (defined in yml): [single_neuron, 2x1_net]")
    required_args.add_argument('-c', '--config-file', **config_file_params)
    # Optional args
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.add_argument('-l', '--log', required=False, default='out.log',
                               help="Name of the output log file")
    optional_args.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    return parser.parse_args()


def main():
    """This is the main function of main.py

    Example:
        python main.py --dataset xor --network 2x1_net --config confs/main_conf.yml
    """

    # Initializing
    get_conf = lambda c_type, c: next(filter(lambda x: x['type'] == c_type, c))
    args = get_args()
    ColorLogger.setup_logger(log_path=args.log, clear_log=True)
    # Load the configurations
    config = Configuration(config_src=args.config_file)
    nn_structures = config.get_config('neural-network')
    nn_structure = get_conf(args.network, nn_structures)
    nn_conf = nn_structure['config']
    nn_type = nn_structure['type']
    datasets = config.get_config('dataset')
    dataset = get_conf(args.dataset, datasets)
    dataset_conf = dataset['config']
    dataset_type = dataset['type']

    # ------- Start of Code ------- #

    inputs = np.array(dataset_conf['inputs'])
    outputs = np.array(dataset_conf['outputs'])
    num_inputs = inputs.shape[1]
    netWork = NeuralNetwork(num_layers=nn_conf['num_layers'],
                            num_neurons=nn_conf['num_neurons'],
                            activations=nn_conf['activations'],
                            num_inputs=num_inputs,
                            loss_function=nn_conf['loss_function'],
                            learning_rate=nn_conf['learning_rate'])
    if args.dataset != 'class_example':
        # Train the network
        logger.nl()  # New line
        logger.info(f'Training the `{nn_type}` network on the `{dataset_type}` dataset')
        for epoch in range(nn_conf['epochs']):
            netWork.train(inputs, outputs)
            loss = netWork.calculate_loss(inputs, outputs)
            if epoch % 200 == 0:
                logger.info(f"Epoch: {epoch} Loss: {loss}")
        # Test on the predictions
        logger.info(f'Predictions on the {dataset_type} dataset', color='cyan')
        for inp, outp in zip(inputs, outputs):
            logger.info(f"True Output: {outp} Prediction: {netWork.calculate(inp)[0]}", color='cyan')
    else:
        # Set up the weights and biases based on the class example
        inputs = inputs[0]
        print(inputs)
        hidden_nodes = np.array(dataset_conf['hidden_nodes'])
        print(hidden_nodes)
        weights = np.array(dataset_conf['weights'])
        print(weights)
        biases = np.array(dataset_conf['biases'])
        print(biases)
        desired_outputs = np.array(dataset_conf['desired_outputs'])
        print(desired_outputs)

    # for i in range(len(inputs)):
    #     print(netWork.calculate(inputs[i]))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
