import argparse
import logging
from sys import exit as sys_exit

from medunda.downloader import configure_parser as downloader_config_parser
from medunda.downloader import downloader
from medunda.plotter import configure_parser as plotter_config_parser
from medunda.plotter import plotter
from medunda.reducer import build_action_args
from medunda.reducer import configure_parser as reducer_config_parser
from medunda.reducer import reducer
from medunda.tools.logging_utils import configure_logger

LOGGER = logging.getLogger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write something here"
    )

    # Create a subparser for each available action, allowing each action
    # to have its own set of command line arguments.
    subparsers = parser.add_subparsers(
        title="tool",
        required=True,
        dest="tool",
        help="Choose which one of the available tools to use"
    )

    subparsers.add_parser(
        "downloader",
        help="Download data and create a Medunda dataset",
    )

    subparsers.add_parser(
        "reducer",
        help="Elaborate downloaded data by performing different statistical operations",
    )

    subparsers.add_parser(
        "plotter",
        help="Plot data from a Medunda dataset",
    )

    downloader_config_parser(subparsers.choices["downloader"])
    reducer_config_parser(subparsers.choices["reducer"])
    plotter_config_parser(subparsers.choices["plotter"])

    return parser.parse_args()


def main():
    configure_logger(LOGGER)
 
    args = parse_args()

    if args.tool == "downloader":
        return downloader(args)
    elif args.tool == "reducer":
        return reducer(
            dataset_path=args.input_dataset,
            output_file=args.output_file,
            action_name=args.action,
            variables=args.variables,
            format=args.format,
            args=build_action_args(args)
        )
    elif args.tool == "plotter":
        return plotter(
            filepath=args.input_file,
            variable=args.variable,
            mode=args.mode,
            args=args
        )


if __name__ == '__main__':
    sys_exit(main())
