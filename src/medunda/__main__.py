import argparse
import logging
from pathlib import Path
from shutil import get_terminal_size
from sys import exit as sys_exit

import tabulate

from medunda.components.frequencies import Frequency
from medunda.components.variables import VariableDataset
from medunda.downloader import configure_parser as downloader_config_parser
from medunda.downloader import downloader
from medunda.plotter import configure_parser as plotter_config_parser
from medunda.plotter import plotter
from medunda.providers import PROVIDERS
from medunda.providers import get_provider
from medunda.reducer import build_action_args
from medunda.reducer import configure_parser as reducer_config_parser
from medunda.reducer import reducer
from medunda.tools.logging_utils import configure_logger

LOGGER = logging.getLogger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write something here")

    # Create a subparser for each available action, allowing each action
    # to have its own set of command line arguments.
    subparsers = parser.add_subparsers(
        title="tool",
        required=True,
        dest="tool",
        help="Choose which one of the available tools to use",
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

    show_subparser = subparsers.add_parser(
        "show",
        help="Show the variables or the providers available in this version of Medunda",
    )

    show_subsubparsers = show_subparser.add_subparsers(
        title="what", required=True, dest="what", help="Choose what to show"
    )

    show_variables_subparser = show_subsubparsers.add_parser(
        "variables",
        help="Show the variables available in this version of Medunda",
    )

    show_variables_subparser.add_argument(
        "--format",
        type=str,
        required=False,
        default="fancy_grid",
        help="The format to use when showing the variables. "
        "It can be any format supported by the tabulate library. "
        "The default is 'fancy_grid'; other common options are "
        "'plain', 'simple', 'grid', 'rst', 'latex', and 'html'. "
        "Others can be found in the tabulate documentation.",
    )

    show_variables_subparser.add_argument(
        "--provider",
        type=str,
        required=False,
        choices=sorted(list(PROVIDERS.keys())),
        default=None,
        help="The provider to filter the variables by. If not "
        "specified, all variables will be shown.",
    )

    show_variables_subparser.add_argument(
        "--provider-config",
        type=Path,
        required=False,
        default=None,
        help="If the provider requires a configuration file, this "
        "argument can be used to specify the path to that file. If no "
        "provider is specified, this argument will be ignored.",
    )

    show_variables_subparser.add_argument(
        "--frequency",
        type=Frequency,
        choices=list(Frequency),
        required=False,
        default=None,
        help="If a provider is specified, this argument can be used to filter the "
        "variables by frequency; in other words, only the variables that the "
        "provider supports at the specified frequency will be shown. "
        "If no provider is specified, this argument will be ignored.",
    )

    show_providers_subparser = show_subsubparsers.add_parser(
        "providers",
        help="Show the providers available in this version of Medunda",
    )

    show_providers_subparser.add_argument(
        "--format",
        type=str,
        required=False,
        default="fancy_grid",
        help="The format to use when showing the providers. "
        "It can be any format supported by the tabulate library. "
        "The default is 'fancy_grid'; other common options are "
        "'plain', 'simple', 'grid', 'rst', 'latex', and 'html'. "
        "Others can be found in the tabulate documentation.",
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
            args=build_action_args(args),
        )
    elif args.tool == "plotter":
        return plotter(
            filepath=args.input_file,
            variable=args.variable,
            mode=args.mode,
            args=args,
        )
    elif args.tool == "show":
        if args.what == "variables":
            if args.provider is None:
                variables = VariableDataset.all_variables()
            else:
                provider = get_provider(args.provider).create(
                    config_file=args.provider_config
                )
                if args.frequency is not None:
                    variables = provider.available_variables(
                        frequency=args.frequency
                    )
                else:
                    variables = VariableDataset()
                    for f in Frequency:
                        f_variables = provider.available_variables(frequency=f)
                        for v in f_variables:
                            variables.add_variable(v)

            table = [[var.name, var.get_label()] for var in variables]
            table.sort(key=lambda x: x[0])
            table_str = tabulate.tabulate(
                table,
                headers=["Variable", "Description"],
                tablefmt=args.format,
            )
            print(table_str)
        elif args.what == "providers":
            table = []
            for provider_name, provider_class in PROVIDERS.items():
                description = provider_class.get_description()
                table.append([provider_name, description])
            table.sort(key=lambda x: x[0])

            column_width = get_terminal_size(fallback=(70, 20)).columns
            first_column_width = max(len(row[0]) for row in table) + 5
            second_column_width = max(
                20, column_width - first_column_width - 5
            )

            table_str = tabulate.tabulate(
                table,
                headers=["Provider", "Description"],
                tablefmt=args.format,
                maxcolwidths=[None, second_column_width],
            )

            print(table_str)
    else:
        LOGGER.error(f"Unknown tool: {args.tool}")
        return 1


if __name__ == "__main__":
    sys_exit(main())
