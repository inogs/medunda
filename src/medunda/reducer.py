
import argparse
import logging
from pathlib import Path


LOGGER = logging.getLogger()


def configure_logger():
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    LOGGER.setLevel(logging.DEBUG)

    logging.getLogger("botocore").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("h5py").setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    LOGGER.addHandler(handler)



def parse_args ():
    """
    parse command line arguments: 
    """
    parser = argparse.ArgumentParser(
        description="elaborate downloaded data by performing different statistical operations"
        )

    parser.add_argument(   
        "--input-file",  
        type=Path,
        required=True,
        help="Path of the input file"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path of the output file"
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=("average", "extract_bottom", "extract_surface"),
        required=True,
        help="End date of the download"
    )

    return parser.parse_args()


def main ():
    args=parse_args()       #parse the command line arguments

    input_file = args.input_file
    output_file = args.output_file
    action = args.action

    match action:
        case "average":
            compute_average(input_file, output_file)
        case "extract_bottom":
            extract_bottom(input_file, output_file)
        case _:
            raise ValueError(
                f"Unexpected action: {action}"
            )



if __name__ == "__main__":
    configure_logger()
    main()
