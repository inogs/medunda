import argparse
from datetime import datetime
import yaml

import copernicusmarine


def argument():
    # TODO: Write a sensible description of what this script does
    parser = argparse.ArgumentParser(
        description="""
        Here there is a description of this program
        """
    )
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        required=True,
        help="""
        The domain of the basin for which the data will be downloaded
        """,
    )
    #
    # parser.add_argument(
    #     "--start-date",
    #     "-s",
    #     type=str,
    #     required=True,
    #     help="""
    #     The start date of the period for which the data will be downloaded
    #     """,
    # )
    #
    # parser.add_argument(
    #     "--end-date",
    #     "-e",
    #     type=str,
    #     required=True,
    #     help="""
    #     The end date of the period for which the data will be downloaded
    #     """,
    # )
    #
    # parser.add_argument(
    #     "--output-dir",
    #     "-o",
    #     type=str,
    #     required=True,
    #     help="""
    #     The path where the data will be downloaded
    #     """,
    # )
    #
    # parser.add_argument(
    #     "--variable",
    #     "-v",
    #     type=str,
    #     required=True,
    #     help="""
    #     Which variable will be downloaded
    #     """
    # )

    return parser.parse_args()


def main():
    args = argument()

    # Read the domain file

    with open(args.domain, "r") as f:
        domain = yaml.safe_load(f)

    copernicusmarine.subset(
        "med-cmcc-tem-rean-d",
        minimum_latitude=domain["domain"]["minimum_latitude"],
        maximum_latitude=domain["domain"]["maximum_latitude"],
        minimum_longitude=domain["domain"]["minimum_longitude"],
        maximum_longitude=domain["domain"]["maximum_longitude"],
        start_datetime=datetime(2020, 1, 1),
        end_datetime=datetime(2020, 1, 31),
        variables=["thetao",],
        output_filename="/dev/shm/test.nc",
    )

    print("Everything has been downloaded")



if __name__ == "__main__":
    main()
