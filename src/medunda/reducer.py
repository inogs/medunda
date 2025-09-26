import argparse
import logging
from pathlib import Path
from typing import Any

from medunda.actions import ActionNotFound
from medunda.actions import averaging_between_layers
from medunda.actions import calculate_stats
from medunda.actions import depth_average
from medunda.actions import extract_bottom
from medunda.actions import extract_extremes
from medunda.actions import extract_layer
from medunda.actions import extract_layer_extremes
from medunda.actions import extract_surface
from medunda.actions import integration
from medunda.components.dataset import read_dataset
from medunda.tools.logging_utils import configure_logger

if __name__ == "__main__":
    LOGGER = logging.getLogger()
else:
    LOGGER = logging.getLogger(__name__)


# This is a list of all the modules that define an action that can be
# executed by the reducer.
ACTION_MODULES = [
    averaging_between_layers,
    calculate_stats,
    depth_average,
    extract_bottom,
    extract_extremes,
    extract_layer,
    extract_layer_extremes,
    extract_surface,
    integration,
    
]

# This is a dictionary that maps the name of an action to the function that
# must be executed. We expect every action module to define a variable named
# `ACTION_NAME` that is the name of the action. This name is both the name of
# the function that will be called and the name of the subparser that will be
# added to the command line parser.
ACTIONS = {
    m.ACTION_NAME : getattr(m, m.ACTION_NAME) for m in ACTION_MODULES
}


def configure_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """
    Parse command line arguments and return parsed arguments object.
    """
    if parser is None:
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
    parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=["netcdf","csv"],
        help="Format of the output-file"
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

    return parser


def build_action_args(args: argparse.Namespace) -> dict[str, Any]:
    # We transform the args object into a dictionary and delete the arguments
    # that are shared among all the actions. In this way, args_values contains
    # only the arguments that are specific to the action that is being executed.
    args_values = dict(**vars(args))
    del args_values["input_dataset"]
    del args_values["output_file"]
    del args_values["format"]
    del args_values["action"]

    if "tool" in args_values:
        del args_values["tool"]

    return args_values


def reducer(dataset_path: Path, output_file:Path, format:str, action_name: str, args: dict):
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
    dataset_result = action(data, **args)

    LOGGER.info('Writing result to "%s" as %s format', output_file, format)

    if format == "netcdf":
        dataset_result.to_netcdf(output_file)
    elif format == "csv":
        dataset_result.to_csv(output_file)
    else:
        raise ValueError(f"{format} is unsupported format for now.")

    return 0


def main():
    configure_logger(LOGGER)

    # parse the command line arguments
    args = configure_parser().parse_args()

    dataset_path = args.input_dataset
    output_file = args.output_file
    format = args.format
    action_name = args.action

    return reducer(
        dataset_path=dataset_path,
        output_file=output_file,
        format=format,
        action_name=action_name,
        args=build_action_args(args)
    )


if __name__ == "__main__":
    main()
