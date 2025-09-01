import argparse
import logging
from sys import exit as sys_exit

from medunda.downloader import downloader
from medunda.downloader import configure_parser as downloader_config_parser
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
        help="Download data from the Medunda dataset",
    )

    downloader_config_parser(subparsers.choices["downloader"])

    return parser.parse_args()


def main():
    configure_logger(LOGGER)
 
    args = parse_args()

    if args.tool != "downloader":
        LOGGER.error(f"Unknown tool {args.action}")
        return 1

    if args.tool == "downloader":
        return downloader(args)


if __name__ == '__main__':
    sys_exit(main())
