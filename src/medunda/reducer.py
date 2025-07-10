import argparse
import logging
from pathlib import Path

import xarray as xr

from medunda.actions import ActionNotFound
from medunda.actions import averaging_between_layers
from medunda.actions import depth_average
from medunda.actions import extract_bottom
from medunda.actions import extract_extremes
from medunda.actions import extract_layer
from medunda.actions import extract_surface
from medunda.actions import extract_layer_extremes
from medunda.actions import calculate_stats
from medunda.dataset import read_dataset
from medunda.tools.logging_utils import configure_logger


LOGGER = logging.getLogger(__name__)

# This is a list of all the modules that define an action that can be
# executed by the reducer.
ACTION_MODULES = [
    averaging_between_layers,
    depth_average,
    extract_bottom,
    extract_extremes,
    extract_layer_extremes,
    extract_layer,
    extract_surface,
    calculate_stats
]

# This is a dictionary that maps the name of an action to the function that
# must be executed. We expect every action module to define a variable named
# `ACTION_NAME` that is the name of the action. This name is both the name of
# the function that will be called and the name of the subparser that will be
# added to the command line parser.
ACTIONS = {
    m.ACTION_NAME : getattr(m, m.ACTION_NAME) for m in ACTION_MODULES
}


def parse_args ():
    """
    Parse command line arguments and return parsed arguments object.
    """
    parser = argparse.ArgumentParser(
        description="elaborate downloaded data by performing different statistical operations"
        )

    parser.add_argument(   
        "--input-dataset",  
        type=Path,
        required=True,
        help="Path of the downloaded Medunda dataset to be processed"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path of the output file"
    )

    # Create a subparser for each available action, allowing each action
    # to have its own set of command line arguments.
    subparsers = parser.add_subparsers(
        title="action",
        required=True,
        dest="action",
        help="Sets which operation must be executed on the input file"
    )

    # Each action module must implement a `configure_parser` function that adds
    # specific arguments to its subparser. This design delegates parser
    # configuration responsibility to individual action modules.
    for module in ACTION_MODULES:
        LOGGER.debug("Letting module %s configure its parser", module.__name__)
        module.configure_parser(subparsers)

    return parser.parse_args()


def reducer(dataset_path: Path, output_file:Path, action_name: str, args: dict):
    if action_name not in ACTIONS:
        valid_action_list = ", ".join(ACTIONS.keys())
        raise ActionNotFound(
            f'Invalid action: "{action_name}". The only allowed actions are {valid_action_list}'
        )

    # Read the dataset file
    LOGGER.info('Reading dataset from "%s"', dataset_path)
    dataset = read_dataset(dataset_path)

    LOGGER.debug("Reading data from the dataset")
    data = dataset.get_data()

    LOGGER.info(
        'Executing action "%s" with the following arguments: %s',
        action_name,
        args
    )

    action = ACTIONS[action_name]
    action(data, output_file, **args)


def main():
    configure_logger(LOGGER)

    # parse the command line arguments
    args=parse_args()

    dataset_path = args.input_dataset
    output_file = args.output_file
    action_name = args.action

    # We transform the args object into a dictionary and delete the arguments
    # that are shared among all the actions. In this way, args_values contains
    # only the arguments that are specific to the action that is being executed.
    args_values = dict(**vars(args))
    del args_values["input_dataset"]
    del args_values["output_file"]
    del args_values["action"]

    reducer(
        dataset_path=dataset_path,
        output_file=output_file,
        action_name=action_name,
        args=args_values
    )


if __name__ == "__main__":
    main()
