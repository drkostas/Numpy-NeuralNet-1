import traceback
import argparse

import numpy as np
from src import Configuration, ColorLogger, Neuron, FullyConnectedLayer, NeuralNetwork

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
    required_args.add_argument('-c', '--config-file', **config_file_params)
    # Optional args
    optional_args = parser.add_argument_group('Optional Arguments')
    optional_args.add_argument('-l', '--log', required=False, default='out.log',
                               help="Name of the output log file")
    optional_args.add_argument('-d', '--debug', action='store_true',
                               help='Enables the debug log messages')
    optional_args.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    return parser.parse_args()


def main():
    """This is the main function of main.py

    Example:
        python main.py -c confs/main_conf.yml
    """

    # Initializing
    args = get_args()
    ColorLogger.setup_logger(log_path=args.log, debug=args.debug, clear_log=True)
    # Load the configuration
    config = Configuration(config_src=args.config_file)
    nn_conf = config.get_config('neural-network')[0]['config']
    dataset_conf = config.get_config('dataset')[0]['config']
    dataset_type = config.get_config('dataset')[0]['type']

    # ------- Start of Code ------- #
    if dataset_type == 'xor':
        inputs = np.array(dataset_conf['inputs'])
        outputs = np.array(dataset_conf['outputs'])
        netWork = NeuralNetwork(num_layers=nn_conf['num_layers'],
                                num_neurons=nn_conf['num_neurons'],
                                activations=nn_conf['activations'],
                                num_inputs=2,
                                loss_function=nn_conf['loss_function'],
                                learning_rate=nn_conf['learning_rate'])
        # test train for xor
        for epoch in range(nn_conf['epochs']):
            netWork.train(inputs, outputs)
            loss = netWork.calculate_loss(inputs, outputs)
            print(f"Epoch: {epoch} Loss: {loss}")

        for inp, outp in zip(inputs, outputs):
            print(netWork.calculate(inp), outp)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(str(e) + '\n' + str(traceback.format_exc()))
        raise e
