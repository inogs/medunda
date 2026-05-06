import argparse
import logging
from pathlib import Path

import cmocean
import pandas as pd
import rioxarray
import xarray as xr

from medunda.components.variables import Variable
from medunda.components.variables import VariableDataset
from medunda.plots import maps
from medunda.plots import timeseries
from medunda.tools.logging_utils import configure_logger

LOGGER = logging.getLogger(__name__)

VAR_METADATA = {
    "o2": {"label": "Oxygen", "unit": "µmol/m³", "cmap": cmocean.cm.deep},  # pyright: ignore[reportAttributeAccessIssue]
    "chl": {
        "label": "Chlorophyll-a",
        "unit": "mg/m³",
        "cmap": cmocean.cm.algae,
    },  # pyright: ignore[reportAttributeAccessIssue]
    "nppv": {
        "label": "Net Primary Production",
        "unit": "mg C/m²/day",
        "cmap": cmocean.cm.matter,
    },  # pyright: ignore[reportAttributeAccessIssue]
    "thetao": {"label": "Temperature", "unit": "°C", "cmap": "coolwarm"},
    "so": {"label": "Salinity", "unit": "PSU", "cmap": "viridis"},
}

DEFAULT_VAR = {
    "unit": "",
    "cmap": "viridis",
}


PLOTS = [
    timeseries,
    maps,
]


def configure_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """
    parse command line arguments:
    --input-file: path of the input file
    --variable: name of the variable to plot
    --output-dir: directory to save the download file
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            description="plots timeseries and maps"
        )

    parser.add_argument(
        "--input-file", type=Path, required=True, help="Path of the input file"
    )
    parser.add_argument(
        "--variable",
        type=str,
        choices=VariableDataset.all_variables().get_variable_names(),
        required=False,
        help="Name of the variable to plot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=False,
        help="Directory where the plots generated are saved",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the plot in an interactive window instead of saving it to a file",
    )

    subparsers = parser.add_subparsers(
        title="mode",
        required=True,
        dest="mode",
        help="Sets which plot must be executed on the input file",
    )

    for mode in PLOTS:
        LOGGER.debug("Letting module %s configure its parser", mode.__name__)
        mode.configure_parser(subparsers)

    return parser


def check_variable(ds, var):
    if isinstance(ds, xr.Dataset):
        if var not in ds:
            raise ValueError(f"Variable '{var}' not found in dataset")
        data = ds[var]

    elif isinstance(ds, xr.DataArray):
        data = ds

    else:
        raise ValueError("Unsupported dataset type")

    if var in VAR_METADATA:
        metadata = VAR_METADATA[var]
    else:
        metadata = DEFAULT_VAR.copy()
        metadata["label"] = var

    return data, metadata


def plotter(filepath: Path, variable: str, mode: str, args):
    """The main function of the plotter module, which is responsible for plotting the data contained in the input file.
    Depending on the mode specified by the user, this function calls the appropriate plotting function to plot the data.
    The plotting functions are defined in the `plots` module and are responsible for plotting timeseries and maps.
    Extracts and plots surface, bottom, and average layers of the given variable."""

    if not filepath.exists():
        raise FileNotFoundError(f"The file '{filepath}' does not exist.")

    if filepath.suffix == ".nc":
        with xr.open_dataset(filepath) as ds:
            data_var, metadata = check_variable(ds, variable)

            if mode == "plotting_timeseries":
                if "time" not in ds[variable].dims:
                    raise ValueError(
                        f"Variable '{variable} does not have the time dimension"
                    )
                else:
                    timeseries.plotting_timeseries(
                        data=data_var,
                        metadata=metadata,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        output_dir=args.output_dir,
                        show_plot=args.show_plot,
                    )

            elif mode == "plotting_maps":
                maps.plotting_maps(
                    data=data_var,
                    metadata=metadata,
                    time=args.time,
                    aggregation_dimension=args.aggregation_dimension,
                    aggregation_method=args.aggregation_method,
                    output_dir=args.output_dir,
                    show_plot=args.show_plot,
                )

            else:
                raise ValueError("Invalid mode")

        return 0

    elif filepath.suffix == ".csv":
        df = pd.read_csv(filepath)

        if "time" in df.columns:
            # Ensure that the time column is in datetime format
            # and set the time column as the index
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df.sort_index(inplace=True)
        else:
            df.set_index(
                [col for col in df.columns if col != variable], inplace=True
            )

        if variable not in df.columns:
            raise ValueError(f"Variable '{variable}' not found in CSV")
        else:
            metadata = DEFAULT_VAR.copy()
            try:
                var_object = Variable.get_by_name(variable)
                metadata["label"] = var_object.get_label()
            except ValueError:
                metadata["label"] = variable

        ds = df.to_xarray()

        data_var = ds[variable]
        data_var, _ = check_variable(ds, variable)

        if mode == "plotting_timeseries":
            if "time" not in ds[variable].dims:
                raise ValueError(
                    f"Variable '{variable} does not have the time dimension"
                )
            else:
                timeseries.plotting_timeseries(
                    data=data_var,
                    metadata=metadata,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    output_dir=args.output_dir,
                    show_plot=args.show_plot,
                )
        elif mode == "plotting_maps":
            raise ValueError("Plotting maps from CSV files is not supported")
        else:
            raise ValueError("Invalid mode")

    elif filepath.suffix == ".tif":
        with rioxarray.open_rasterio(filepath) as ds:
            data_var, metadata = check_variable(ds, variable)

            if mode == "plotting_timeseries":
                raise ValueError(
                    "Plotting timeseries from GeoTIFF files is not supported"
                )

            elif mode == "plotting_maps":
                maps.plotting_maps(
                    data=data_var,
                    metadata=metadata,
                    time=args.time,
                    aggregation_dimension=args.aggregation_dimension,
                    aggregation_method=args.aggregation_method,
                    output_dir=args.output_dir,
                    show_plot=args.show_plot,
                )
            else:
                raise ValueError("Invalid mode")

    else:
        raise ValueError(
            f"Unsupported file format '{filepath.suffix}'. Supported formats are: .nc, .csv, .tif"
        )


def main():
    args = configure_parser().parse_args()
    configure_logger(LOGGER)

    variable = args.variable
    mode = args.mode
    data_file = args.input_file
    output_dir = args.output_dir
    show_plot = args.show_plot

    LOGGER.info(f"Selected file: {data_file.name}")

    plotter(
        filepath=data_file,
        variable=variable,
        mode=mode,
        args=args,
        output_dir=output_dir,
        show_plot=show_plot,
    )

    LOGGER.info(
        f"Plotting completed for variable '{variable}' in mode '{mode}'"
    )


if __name__ == "__main__":
    main()
